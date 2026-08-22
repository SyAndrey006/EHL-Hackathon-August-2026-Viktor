#!/usr/bin/env python3
"""Explainable off-policy evaluation by nearest-neighbour trajectory matching.

The logged model is the only observed treatment: a trajectory has no outcome for
the models that did not serve it.  We therefore compare outcomes only between
trajectories that look similar *before reading their recovered execution trace*.
For every trajectory served by model A, find the closest trajectory served by
model B on prompt/input size and tool availability.  The mean difference in the
structural outcome proxy is our estimate of A minus B for that observed task mix.

This is deliberately matching, rather than a black-box causal model.  It does
not manufacture a counterfactual for a model pair with no sufficiently similar
logged tasks.

Usage:
    python scripts/offpolicy_eval.py export/
Writes results/offpolicy_pairs.csv and results/offpolicy_summary.json.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from load_trajectories import est_tokens, first_user_text, group_trajectories, iter_requests
from outcome_signal import cohort_expected_calls, cohort_key, estimate_outcome


# Only these task descriptors enter matching.  They are all present on the first
# request except n_calls, included here because the challenge explicitly permits
# trajectory shape for OPE.  Do NOT use n_calls as a feature for the first-call
# router: it would leak future behaviour into a deployment decision.
MATCH_NUMERIC = ("first_user_chars", "first_input_tokens", "tool_count", "n_calls")


def tool_names(calls):
    return frozenset(t.get("name", "") for t in calls[0].get("tools", []) if t.get("name"))


def trajectory_features(calls):
    """Observable task/shape fields used by the matching estimator."""
    return {
        "first_user_chars": len(first_user_text(calls[0])),
        "first_input_tokens": est_tokens(calls[0]["input"]),
        "tool_count": len(tool_names(calls)),
        "n_calls": len(calls),
        "tool_set": tool_names(calls),
    }


def robust_scales(records):
    """Interquartile ranges; avoids a single giant prompt dominating distance."""
    scales = {}
    for name in MATCH_NUMERIC:
        values = sorted(r["features"][name] for r in records)
        q1 = values[len(values) // 4]
        q3 = values[(3 * len(values)) // 4]
        scales[name] = max(1.0, float(q3 - q1))
    return scales


def match_distance(a, b, scales):
    """Mean normalized numeric difference + tool-set Jaccard distance, [0, inf)."""
    numeric = sum(abs(a["features"][n] - b["features"][n]) / scales[n] for n in MATCH_NUMERIC)
    left, right = a["features"]["tool_set"], b["features"]["tool_set"]
    union = left | right
    tool_distance = 0.0 if not union else 1.0 - len(left & right) / len(union)
    return numeric / len(MATCH_NUMERIC) + tool_distance


def build_records(groups):
    """Attach an observed outcome proxy and logged model to each trajectory."""
    expected = cohort_expected_calls(groups)
    records = []
    for key, calls in groups.items():
        model = calls[0]["model"]
        if any(c["model"] != model for c in calls):
            # The premise says this cannot happen. Excluding violations avoids
            # assigning one treatment to a trajectory that received several.
            continue
        records.append({
            "trajectory": key,
            "model": model,
            "outcome": estimate_outcome(calls, expected[cohort_key(calls)]),
            "features": trajectory_features(calls),
        })
    return records


def pairwise_matching(records, caliper=1.25):
    """Return A-vs-B effects by matching every A trajectory to closest B trace.

    A pair is included only when its normalized distance <= caliper.  This
    turns lack of common support into a visible smaller sample rather than a
    spurious precise answer.
    """
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    scales = robust_scales(records)
    rows = []
    models = sorted(by_model)
    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            matched = []
            for a in by_model[model_a]:
                candidates = [(match_distance(a, b, scales), b) for b in by_model[model_b]]
                distance, b = min(candidates, key=lambda x: x[0])
                if distance <= caliper:
                    matched.append((a, b, distance))
            if not matched:
                rows.append({"model_a": model_a, "model_b": model_b,
                             "n_a": len(by_model[model_a]), "n_b": len(by_model[model_b]),
                             "matched": 0, "mean_distance": "", "outcome_a": "",
                             "outcome_b_matched": "", "effect_a_minus_b": "",
                             "standard_error": "", "support": "none"})
                continue
            effects = [a["outcome"] - b["outcome"] for a, b, _ in matched]
            mean = sum(effects) / len(effects)
            variance = (sum((x - mean) ** 2 for x in effects) / (len(effects) - 1)
                        if len(effects) > 1 else 0.0)
            rows.append({
                "model_a": model_a, "model_b": model_b,
                "n_a": len(by_model[model_a]), "n_b": len(by_model[model_b]),
                "matched": len(matched),
                "mean_distance": round(sum(d for _, _, d in matched) / len(matched), 4),
                "outcome_a": round(sum(a["outcome"] for a, _, _ in matched) / len(matched), 4),
                "outcome_b_matched": round(sum(b["outcome"] for _, b, _ in matched) / len(matched), 4),
                "effect_a_minus_b": round(mean, 4),
                "standard_error": round(math.sqrt(variance / len(effects)), 4),
                "support": "matched",
            })
    return rows, scales


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    caliper = float(sys.argv[2]) if len(sys.argv) > 2 else 1.25
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    records = build_records(groups)
    if len(records) < 2 or len({r["model"] for r in records}) < 2:
        raise SystemExit("need trajectories from at least two single-model groups for off-policy matching")
    rows, scales = pairwise_matching(records, caliper)
    Path("results").mkdir(exist_ok=True)
    with open("results/offpolicy_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "method": "one-way nearest-neighbour matching: model A trajectory -> closest model B trajectory",
        "outcome": "structural execution-health proxy from outcome_signal.py",
        "caliper": caliper,
        "distance_scales_iqr": scales,
        "records": len(records),
        "models": sorted({r["model"] for r in records}),
        "bias_warnings": [
            "Harder tasks may have been deliberately routed to stronger models. Matching only controls listed observables, so unobserved difficulty can make a strong model look necessary.",
            "n_calls is post-treatment trajectory shape; it is allowed here as a descriptive challenge feature but can absorb model behaviour and bias causal comparisons. Re-run without it as a sensitivity check.",
            "Nearest neighbours may be reused, estimates are asymmetric (A task mix), and small matched counts or large distances mean weak overlap.",
            "The outcome is a proxy for visible execution health, not final-answer correctness; its own failure modes are documented in outcome_signal.py.",
            "This evaluates outcomes only. Counterfactual costs are separately computed by cost_model.py using estimated input tokens and inferred cache overlap; final output costs remain unobserved."
        ],
    }
    Path("results/offpolicy_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"eligible single-model trajectories={len(records)}; models={len(summary['models'])}; caliper={caliper}")
    print("pairwise effect = outcome(model_a) - outcome(matched model_b)")
    for row in rows:
        print(f"{row['model_a']} vs {row['model_b']}: matched={row['matched']} "
              f"effect={row['effect_a_minus_b']} mean_distance={row['mean_distance']} support={row['support']}")
    print("wrote results/offpolicy_pairs.csv and results/offpolicy_summary.json")
    print("BIAS WARNING: matching does not remove unobserved task difficulty or proxy-label error.")


if __name__ == "__main__":
    main()
