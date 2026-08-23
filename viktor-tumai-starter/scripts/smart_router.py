#!/usr/bin/env python3
"""Predictive, cache-aware router with a one-way cheap -> logged-model upgrade.

The lightweight logistic model is trained on the observable action-shape proxy. Its
prediction adjusts (rather than replaces) the matched cheap-tier trajectory-success
prior from outcome_evaluator.py. This distinction matters: proxy_simple is not a
ground-truth cheap-model success label.

Usage: python scripts/smart_router.py export/ results/call_features.csv
Writes results/routes_smart.jsonl
"""
from __future__ import annotations

import argparse, csv, json, math, statistics, sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from cost_model import load_pricing, logged_route, trajectory_cost
from load_trajectories import group_trajectories, iter_requests
from outcome_evaluator import configure_estimator, estimate_route_quality

CHEAP = {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-luna"}
SAFETY_THRESHOLD = 0.85
SIGNAL_WEIGHT = 0.75


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value); return 1 / (1 + z)
    z = math.exp(value); return z / (1 + z)


def logit(probability):
    p = min(1 - 1e-6, max(1e-6, probability))
    return math.log(p / (1 - p))


def cheap_for(model):
    return CHEAP["claude"] if model.startswith("claude") else CHEAP["gpt"]


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


class WeightedLogistic:
    """Small dependency-free logistic classifier with L2 regularization."""
    numeric = ("has_reasoning", "has_input_image", "position_fraction")

    def fit(self, rows):
        train = [row for row in rows if row["outcome_observed"] == "1" and row["proxy_simple"] != ""]
        self.categories = sorted({row["task_category"] for row in rows})
        self.means = {name: statistics.fmean(float(row[name]) for row in train) for name in self.numeric}
        self.scales = {name: max(0.1, statistics.pstdev(float(row[name]) for row in train)) for name in self.numeric}
        self.names = ["intercept", *self.numeric, *(f"category={c}" for c in self.categories)]
        self.weights = [0.0] * len(self.names)
        # Multipliers reduce the effective regularization penalty on the four signals
        # requested by the policy design, making them dominate incidental features.
        self.multipliers = [1.0, 2.5, 2.5, 2.0, *([1.75] * len(self.categories))]
        y = [int(row["proxy_simple"]) for row in train]
        self.prior = statistics.fmean(y)
        self.weights[0] = logit(self.prior)
        rate, l2 = 0.08, 0.08
        for _ in range(1200):
            gradient = [0.0] * len(self.weights)
            for row, label in zip(train, y):
                x = self.vector(row)
                error = sigmoid(sum(w*v for w, v in zip(self.weights, x))) - label
                for j, value in enumerate(x): gradient[j] += error * value
            for j in range(len(self.weights)):
                penalty = 0.0 if j == 0 else l2 * self.weights[j]
                self.weights[j] -= rate * (gradient[j] / len(train) + penalty)
        return self

    def vector(self, row):
        values = [1.0]
        values.extend((float(row[name]) - self.means[name]) / self.scales[name] for name in self.numeric)
        values.extend(1.0 if row["task_category"] == category else 0.0 for category in self.categories)
        return [value * multiplier for value, multiplier in zip(values, self.multipliers)]

    def predict(self, row):
        return sigmoid(sum(w*v for w, v in zip(self.weights, self.vector(row))))


def calibrated_cheap_success(model, row, trajectory, n_calls, classifier):
    """Adjust matched cheap-tier success by relative action-complexity evidence."""
    matched = estimate_route_quality(trajectory, [model] * n_calls)
    proxy_probability = classifier.predict(row)
    probability = sigmoid(logit(matched) + SIGNAL_WEIGHT * (logit(proxy_probability) - logit(classifier.prior)))
    return probability, matched, proxy_probability


def route_trajectory(trajectory, calls, feature_rows, classifier, threshold):
    logged = calls[0]["model"]
    cheap = cheap_for(logged)
    upgraded, route, scores = False, [], []
    for row in feature_rows:
        probability, matched, proxy = calibrated_cheap_success(cheap, row, trajectory, len(calls), classifier)
        if not upgraded and probability < threshold:
            upgraded = True
        route.append(logged if upgraded else cheap)
        scores.append((probability, matched, proxy))
    return route, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("features", nargs="?", default="results/call_features.csv")
    parser.add_argument("--outcomes", default="results/outcomes.json")
    parser.add_argument("--output", default="results/routes_smart.jsonl")
    parser.add_argument("--threshold", type=float, default=SAFETY_THRESHOLD)
    args = parser.parse_args()
    rows = load_rows(args.features)
    classifier = WeightedLogistic().fit(rows)
    configure_estimator(args.outcomes, args.features)
    by_trajectory = {}
    for row in rows:
        by_trajectory.setdefault(row["trajectory"], []).append(row)
    for trajectory in by_trajectory:
        by_trajectory[trajectory].sort(key=lambda row: int(row["call_index"]))
    groups = group_trajectories(req for _, _, req in iter_requests(args.export))
    pricing, records = load_pricing(), []
    for trajectory, calls in groups.items():
        route, scores = route_trajectory(trajectory, calls, by_trajectory[trajectory], classifier, args.threshold)
        c_logged, _ = trajectory_cost(calls, logged_route(calls), pricing)
        c_routed, _ = trajectory_cost(calls, route, pricing)
        records.append({"trajectory": trajectory, "n_calls": len(calls),
                        "logged_model": calls[0]["model"], "route": route,
                        "cost_logged_usd": round(c_logged, 6), "cost_routed_usd": round(c_routed, 6),
                        "switches": sum(route[i] != route[i-1] for i in range(1, len(route)))})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record) + "\n")
    total_logged = sum(r["cost_logged_usd"] for r in records)
    total_routed = sum(r["cost_routed_usd"] for r in records)
    upgraded = sum(any(m == r["logged_model"] for m in r["route"]) and r["route"][0] != r["logged_model"] for r in records)
    start_premium = sum(r["route"][0] == r["logged_model"] for r in records)
    switches = Counter(r["switches"] for r in records)
    print(f"trained proxy classifier: n={sum(r['outcome_observed']=='1' for r in rows)} prior_simple={classifier.prior:.1%}")
    print(f"threshold={args.threshold:.2f} trajectories={len(records)} start_premium={start_premium} upgrade_later={upgraded}")
    print(f"switch distribution={dict(sorted(switches.items()))} (invariant: only 0 or 1)")
    print(f"logged cost=${total_logged:.4f} smart cost=${total_routed:.4f} ({total_routed/total_logged-1:+.1%})")
    print("WARNING: classifier target is action-shape proxy, calibrated with sparse matched outcomes; not causal proof.")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
