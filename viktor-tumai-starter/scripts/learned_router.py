#!/usr/bin/env python3
"""Interpretable learned whole-trajectory model router.

Train a depth-three regression tree on the structural outcome proxy observed in
logged trajectories.  At routing time it evaluates each observed candidate
model and selects the cheapest predicted to retain proxy outcome.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from cost_model import load_pricing, price_of, trajectory_cost, logged_route
from load_trajectories import est_tokens, first_user_text, group_trajectories, iter_requests
from outcome_signal import cohort_expected_calls, cohort_key, estimate_outcome


MAX_DEPTH = 3
MIN_LEAF = 6
# Two observations lets the synthetic smoke sample exercise candidate routes;
# the confidence router subsequently applies its stricter local-support gate.
MIN_MODEL_SUPPORT = 2
QUALITY_TOLERANCE = 0.05
MIN_PREDICTED_OUTCOME = 0.65


def first_call_features(calls):
    """Features available before the first response; no later-call leakage."""
    first = calls[0]
    user = first_user_text(first)
    tools = first.get("tools", [])
    content = " ".join(json.dumps(i, ensure_ascii=False) for i in first.get("input", []))
    return {
        "first_user_chars": float(len(user)),
        "first_input_tokens": float(est_tokens(first["input"])),
        "tool_count": float(sum(bool(t.get("name")) for t in tools)),
        "has_image": float("input_image" in content),
    }


def candidate_features(base, candidate, models):
    """Explicit one-hot candidate-model flags make every tree split readable."""
    x = dict(base)
    for model in models:
        x[f"candidate_is_{model}"] = float(candidate == model)
    return x


def sse(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values)


def fit_tree(rows, feature_names, depth=0):
    """Fit a tiny CART-style regression tree using squared-error splits."""
    y = [row["outcome"] for row in rows]
    node = {"n": len(rows), "mean": sum(y) / len(y)}
    if depth >= MAX_DEPTH or len(rows) < 2 * MIN_LEAF or len(set(y)) == 1:
        return node
    parent_error = sse(y)
    best = None
    for feature in feature_names:
        ordered = sorted(rows, key=lambda row: row["x"][feature])
        for i in range(MIN_LEAF, len(ordered) - MIN_LEAF + 1):
            left, right = ordered[:i], ordered[i:]
            lo, hi = left[-1]["x"][feature], right[0]["x"][feature]
            if lo == hi:
                continue
            error = sse([row["outcome"] for row in left]) + sse([row["outcome"] for row in right])
            gain = parent_error - error
            if best is None or gain > best[0]:
                best = (gain, feature, (lo + hi) / 2.0, left, right)
    if best is None or best[0] <= 1e-12:
        return node
    _, feature, threshold, left, right = best
    node.update({"feature": feature, "threshold": threshold,
                 "left": fit_tree(left, feature_names, depth + 1),
                 "right": fit_tree(right, feature_names, depth + 1)})
    return node


def predict(tree, x):
    while "feature" in tree:
        tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["mean"]


def tree_rules(tree, indent=""):
    """Plain-text rules stored alongside routes for review/presentation."""
    if "feature" not in tree:
        return [f"{indent}predict {tree['mean']:.3f} (n={tree['n']})"]
    label = f"{tree['feature']} <= {tree['threshold']:.3f}"
    return ([f"{indent}if {label}:"] + tree_rules(tree["left"], indent + "  ") +
            [f"{indent}else:"] + tree_rules(tree["right"], indent + "  "))


def training_rows(groups, models):
    """Observed treatment/outcome rows. Labels use full trace only in training."""
    expected = cohort_expected_calls(groups)
    rows = []
    for calls in groups.values():
        model = calls[0]["model"]
        if any(call["model"] != model for call in calls):
            continue
        rows.append({"x": candidate_features(first_call_features(calls), model, models),
                     "outcome": estimate_outcome(calls, expected[cohort_key(calls)]),
                     "model": model})
    return rows


def route_trajectory(calls, tree, models, model_counts, pricing):
    """Pick the lowest-input-price candidate predicted to retain proxy outcome."""
    logged = calls[0]["model"]
    base = first_call_features(calls)
    logged_prediction = predict(tree, candidate_features(base, logged, models))
    supported = [model for model in models if model_counts[model] >= MIN_MODEL_SUPPORT]
    viable = []
    for candidate in supported:
        predicted = predict(tree, candidate_features(base, candidate, models))
        if predicted >= MIN_PREDICTED_OUTCOME and predicted >= logged_prediction - QUALITY_TOLERANCE:
            viable.append((price_of(candidate, pricing)[0], candidate, predicted))
    if not viable:
        return [logged] * len(calls), logged_prediction, logged_prediction, "no_supported_safe_candidate"
    _, chosen, chosen_prediction = min(viable)
    return [chosen] * len(calls), logged_prediction, chosen_prediction, "predicted_safe"


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/routes.jsonl")
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    models = sorted({calls[0]["model"] for calls in groups.values()
                     if calls and len({call["model"] for call in calls}) == 1})
    rows = training_rows(groups, models)
    if len(rows) < 2 * MIN_LEAF:
        raise SystemExit(f"need at least {2 * MIN_LEAF} single-model trajectories to fit the learned router")
    features = list(rows[0]["x"])
    tree = fit_tree(rows, features)
    counts = Counter(row["model"] for row in rows)
    pricing = load_pricing()
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_logged = total_routed = 0.0
    decisions = Counter()
    with destination.open("w") as out:
        for key, calls in groups.items():
            if len({call["model"] for call in calls}) != 1:
                continue
            logged = logged_route(calls)
            routed, logged_prediction, route_prediction, reason = route_trajectory(
                calls, tree, models, counts, pricing)
            cost_logged, _ = trajectory_cost(calls, logged, pricing)
            cost_routed, _ = trajectory_cost(calls, routed, pricing)
            total_logged += cost_logged
            total_routed += cost_routed
            decisions[reason] += 1
            out.write(json.dumps({
                "trajectory": key, "n_calls": len(calls), "logged_model": logged[0], "route": routed,
                "cost_logged_usd": round(cost_logged, 6), "cost_routed_usd": round(cost_routed, 6),
                "switches": 0, "predicted_logged_outcome": round(logged_prediction, 4),
                "predicted_routed_outcome": round(route_prediction, 4), "route_reason": reason,
            }) + "\n")
    Path("results/learned_tree_rules.txt").write_text("\n".join(tree_rules(tree)) + "\n")
    print(f"trained depth<={MAX_DEPTH} regression tree on {len(rows)} trajectories; per-model={dict(counts)}")
    print("learned rules written to results/learned_tree_rules.txt")
    print(f"logged cost (est. input tokens, official prices): ${total_logged:,.4f}")
    print(f"routed cost: ${total_routed:,.4f} ({(total_routed / total_logged - 1):+.1%}, cache-aware)")
    print(f"decisions: {dict(decisions)}")
    print("NOTE: outcome retention is model-predicted from observational proxy labels; use offpolicy_eval.py for matching diagnostics.")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
