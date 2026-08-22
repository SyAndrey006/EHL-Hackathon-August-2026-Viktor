#!/usr/bin/env python3
"""Evidence-Gated Confidence Router.

The learned regression tree is a *proposal* engine: it suggests a cheaper
whole-trajectory model from observational proxy labels.  This module makes the
proposal operationally safe by requiring local, pre-treatment evidence before
switching models.  If comparable historical examples are sparse, far away, or
too variable, it abstains and keeps the logged model for the whole trajectory.

Only features available before the first response are used for the gate.  In
particular, ``n_calls`` is intentionally excluded: it is known only after a
trajectory has run and would leak future information into a deployment policy.

Usage:
    python scripts/confidence_router.py export/
    python scripts/confidence_router.py export/ results/confidence_routes.jsonl

All outcome estimates remain observational execution-health proxies.  A lower
bootstrap bound is a variance guardrail, not proof of causal model quality.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from cost_model import load_pricing, trajectory_cost, logged_route
from learned_router import (
    MIN_LEAF,
    first_call_features,
    fit_tree,
    route_trajectory as propose_route,
    training_rows,
    tree_rules,
)
from load_trajectories import group_trajectories, iter_requests
from outcome_signal import cohort_expected_calls, cohort_key, estimate_outcome


# Conservative defaults for the small-data setting.  They are command-line
# parameters so the team can pre-register sensible sensitivity runs rather than
# silently changing the policy after inspecting its apparent savings.
MIN_LOCAL_SUPPORT = 3
MAX_NEIGHBORS = 5
MATCH_CALIPER = 1.25
BOOTSTRAP_SAMPLES = 1_000
LOWER_QUANTILE = 0.10


def deployment_features(calls):
    """Return exactly the fields available before the first model response.

    ``first_call_features`` comes from the learned proposal engine, ensuring
    the tree and gate use the same prompt-size/tool-count/image definitions.
    The gate additionally needs a tool-name set to compute Jaccard distance.
    It does *not* include n_calls, history length after call one, outcomes, or
    any recovered tool result.
    """
    base = first_call_features(calls)
    tool_set = frozenset(
        tool.get("name", "") for tool in calls[0].get("tools", []) if tool.get("name")
    )
    return {
        "first_user_chars": base["first_user_chars"],
        "first_input_tokens": base["first_input_tokens"],
        "tool_count": base["tool_count"],
        "has_image": base["has_image"],
        "tool_set": tool_set,
    }


MATCH_NUMERIC = ("first_user_chars", "first_input_tokens", "tool_count", "has_image")


def robust_scales(records):
    """IQR scales stop a single very long prompt dominating all distances."""
    scales = {}
    for name in MATCH_NUMERIC:
        values = sorted(record["features"][name] for record in records)
        q1 = values[len(values) // 4]
        q3 = values[(3 * len(values)) // 4]
        scales[name] = max(1.0, float(q3 - q1))
    return scales


def match_distance(left, right, scales):
    """Distance on pre-treatment features only; lower means more comparable."""
    numeric = sum(
        abs(left["features"][name] - right["features"][name]) / scales[name]
        for name in MATCH_NUMERIC
    ) / len(MATCH_NUMERIC)
    left_tools, right_tools = left["features"]["tool_set"], right["features"]["tool_set"]
    union = left_tools | right_tools
    tool_distance = 0.0 if not union else 1.0 - len(left_tools & right_tools) / len(union)
    return numeric + tool_distance


def build_records(groups):
    """Create one observed-treatment record per valid, single-model task."""
    expected_calls = cohort_expected_calls(groups)
    records = []
    for trajectory, calls in groups.items():
        if not calls or len({call["model"] for call in calls}) != 1:
            continue
        records.append({
            "trajectory": trajectory,
            "model": calls[0]["model"],
            "features": deployment_features(calls),
            # This label is used only as historical evidence.  It is never a
            # feature presented to the router at decision time.
            "outcome": estimate_outcome(calls, expected_calls[cohort_key(calls)]),
        })
    return records


def nearest_neighbours(target, records, model, scales, caliper, max_neighbours):
    """Return closest same-model historical records inside the support caliper.

    During this offline replay, excluding the target trajectory gives the gate
    the same information it would have for a genuinely new task.  It also
    prevents the logged trajectory from certifying its own quality.
    """
    candidates = []
    for record in records:
        if record["trajectory"] == target["trajectory"] or record["model"] != model:
            continue
        distance = match_distance(target, record, scales)
        if distance <= caliper:
            candidates.append((distance, record))
    candidates.sort(key=lambda item: (item[0], item[1]["trajectory"]))
    return candidates[:max_neighbours]


def weighted_mean(neighbours):
    """Inverse-distance outcome mean; a tiny ridge avoids infinite weights."""
    if not neighbours:
        return None
    weights = [1.0 / (distance + 0.05) for distance, _ in neighbours]
    return sum(weight * record["outcome"] for weight, (_, record) in zip(weights, neighbours)) / sum(weights)


def _seed_for(*parts):
    """Stable seeds make bootstrap outputs reproducible across Python sessions."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_effect(candidate_neighbours, logged_neighbours, samples, quantile, seed):
    """Bootstrap the local outcome difference: candidate minus logged model.

    Each replicate resamples the two local neighbour sets with replacement,
    retaining inverse-distance weighting.  The requested lower quantile is a
    conservative quality-retention bound: we only switch if even this bound
    permits the allowed degradation.
    """
    if not candidate_neighbours or not logged_neighbours:
        return None, None
    point_effect = weighted_mean(candidate_neighbours) - weighted_mean(logged_neighbours)
    rng = random.Random(seed)

    def resample_weighted_mean(neighbours):
        draw = [neighbours[rng.randrange(len(neighbours))] for _ in neighbours]
        return weighted_mean(draw)

    effects = sorted(
        resample_weighted_mean(candidate_neighbours) - resample_weighted_mean(logged_neighbours)
        for _ in range(samples)
    )
    index = max(0, min(len(effects) - 1, math.floor((len(effects) - 1) * quantile)))
    return point_effect, effects[index]


def gate_proposal(target, proposal, records, scales, args):
    """Accept a tree switch only when both treatment groups have local support."""
    logged = target["model"]
    artifact = {
        "tree_proposal": proposal,
        "candidate_support": 0,
        "logged_support": 0,
        "candidate_mean_distance": None,
        "logged_mean_distance": None,
        "local_effect_candidate_minus_logged": None,
        "lower_confidence_bound": None,
        # QUALITY_TOLERANCE is an allowed *drop*, so the effect threshold is
        # negative: -0.05 means candidate may be at most .05 worse.
        "required_lower_bound": -args.quality_tolerance,
    }
    if proposal == logged:
        artifact.update({"gate_decision": "kept_logged", "fallback_reason": "tree_kept_logged"})
        return logged, artifact

    candidate_neighbours = nearest_neighbours(
        target, records, proposal, scales, args.caliper, args.max_neighbours)
    logged_neighbours = nearest_neighbours(
        target, records, logged, scales, args.caliper, args.max_neighbours)
    artifact.update({
        "candidate_support": len(candidate_neighbours),
        "logged_support": len(logged_neighbours),
        "candidate_mean_distance": round(sum(d for d, _ in candidate_neighbours) / len(candidate_neighbours), 4)
        if candidate_neighbours else None,
        "logged_mean_distance": round(sum(d for d, _ in logged_neighbours) / len(logged_neighbours), 4)
        if logged_neighbours else None,
    })
    if len(candidate_neighbours) < args.min_support or len(logged_neighbours) < args.min_support:
        artifact.update({"gate_decision": "rejected", "fallback_reason": "insufficient_local_support"})
        return logged, artifact

    effect, lower_bound = bootstrap_effect(
        candidate_neighbours,
        logged_neighbours,
        args.bootstrap_samples,
        args.lower_quantile,
        _seed_for(target["trajectory"], logged, proposal),
    )
    artifact.update({
        "local_effect_candidate_minus_logged": round(effect, 4),
        "lower_confidence_bound": round(lower_bound, 4),
    })
    if lower_bound < -args.quality_tolerance:
        artifact.update({"gate_decision": "rejected", "fallback_reason": "lower_bound_below_quality_tolerance"})
        return logged, artifact
    artifact.update({"gate_decision": "accepted", "fallback_reason": None})
    return proposal, artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Gate learned routes with local OPE evidence.")
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("destination", nargs="?", default="results/confidence_routes.jsonl")
    parser.add_argument("--min-support", type=int, default=MIN_LOCAL_SUPPORT)
    parser.add_argument("--max-neighbours", type=int, default=MAX_NEIGHBORS)
    parser.add_argument("--caliper", type=float, default=MATCH_CALIPER)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--lower-quantile", type=float, default=LOWER_QUANTILE)
    # Import the proposal engine's definition so policy meanings cannot drift.
    from learned_router import QUALITY_TOLERANCE
    parser.add_argument("--quality-tolerance", type=float, default=QUALITY_TOLERANCE)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_support < 1 or args.max_neighbours < args.min_support:
        raise SystemExit("max-neighbours must be >= min-support >= 1")
    if args.bootstrap_samples < 1 or not 0.0 < args.lower_quantile < 1.0:
        raise SystemExit("bootstrap-samples must be positive and lower-quantile must be in (0, 1)")

    groups = group_trajectories(request for _, _, request in iter_requests(args.export))
    records = build_records(groups)
    if len(records) < 2 * MIN_LEAF:
        raise SystemExit(f"need at least {2 * MIN_LEAF} valid single-model trajectories to fit the proposal tree")

    models = sorted({record["model"] for record in records})
    rows = training_rows(groups, models)
    if not rows:
        raise SystemExit("no valid single-model trajectories available for learned routing")
    tree = fit_tree(rows, list(rows[0]["x"]))
    model_counts = Counter(row["model"] for row in rows)
    scales = robust_scales(records)
    pricing = load_pricing()
    by_trajectory = {record["trajectory"]: record for record in records}

    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    decisions = Counter()
    total_logged = total_routed = 0.0
    with destination.open("w") as out:
        for trajectory, calls in groups.items():
            if trajectory not in by_trajectory:
                continue
            logged_route_models = logged_route(calls)
            proposed_route, logged_prediction, proposal_prediction, proposal_reason = propose_route(
                calls, tree, models, model_counts, pricing)
            proposal = proposed_route[0]
            final_model, gate = gate_proposal(by_trajectory[trajectory], proposal, records, scales, args)
            final_route = [final_model] * len(calls)  # whole-trajectory routing: zero switches
            logged_cost, _ = trajectory_cost(calls, logged_route_models, pricing)
            routed_cost, _ = trajectory_cost(calls, final_route, pricing)
            total_logged += logged_cost
            total_routed += routed_cost
            decisions[gate["gate_decision"]] += 1
            out.write(json.dumps({
                "trajectory": trajectory,
                "n_calls": len(calls),
                "logged_model": logged_route_models[0],
                "tree_proposal": proposal,
                "final_route": final_route,
                "route": final_route,  # compatibility with existing result readers
                "gate_decision": gate["gate_decision"],
                "fallback_reason": gate["fallback_reason"],
                "proposal_reason": proposal_reason,
                "predicted_logged_outcome": round(logged_prediction, 4),
                "predicted_proposal_outcome": round(proposal_prediction, 4),
                "candidate_support": gate["candidate_support"],
                "logged_support": gate["logged_support"],
                "candidate_mean_distance": gate["candidate_mean_distance"],
                "logged_mean_distance": gate["logged_mean_distance"],
                "local_effect_candidate_minus_logged": gate["local_effect_candidate_minus_logged"],
                "lower_confidence_bound": gate["lower_confidence_bound"],
                "required_lower_bound": gate["required_lower_bound"],
                "cost_logged_usd": round(logged_cost, 6),
                "cost_routed_usd": round(routed_cost, 6),
                "switches": 0,
            }) + "\n")

    rules_path = destination.parent / "confidence_tree_rules.txt"
    rules_path.write_text("\n".join(tree_rules(tree)) + "\n")
    print(f"trained learned proposal tree on {len(rows)} trajectories; per-model={dict(model_counts)}")
    print(f"confidence gate: min_support={args.min_support}, caliper={args.caliper}, "
          f"bootstrap={args.bootstrap_samples}, lower_quantile={args.lower_quantile}")
    print(f"decisions: {dict(decisions)}")
    print(f"logged cost (est. input tokens, official prices): ${total_logged:,.4f}")
    print(f"confidence-routed cost: ${total_routed:,.4f} "
          f"({(total_routed / total_logged - 1):+.1%}, cache-aware)")
    print(f"wrote {destination} and {rules_path}")
    print("NOTE: accepted switches have local proxy support, not experimentally proven correctness.")


if __name__ == "__main__":
    main()
