#!/usr/bin/env python3
"""Compare smart and baseline matched cost-quality frontiers fairly.

Both proposals are ranked by expected quality loss, then savings, before sweeping
adoption. This avoids claiming an elbow improvement from different sweep ordering.
"""
import argparse, csv, json, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from outcome_evaluator import configure_estimator, estimate_route_quality


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def curve(records, policy):
    candidates = []
    for rec in records:
        logged_route = [rec["logged_model"]] * rec["n_calls"]
        logged_quality = estimate_route_quality(rec["trajectory"], logged_route)
        routed_quality = estimate_route_quality(rec["trajectory"], rec["route"])
        loss = max(0.0, logged_quality - routed_quality)
        savings = rec["cost_logged_usd"] - rec["cost_routed_usd"]
        candidates.append((loss, -savings, rec, logged_quality, routed_quality))
    candidates.sort(key=lambda item: (item[0], item[1]))
    rows = []
    for frac in [i / 10 for i in range(11)]:
        adopted = {id(item[2]) for item in candidates[:int(round(frac * len(candidates)))]}
        cost = sum(rec["cost_routed_usd"] if id(rec) in adopted else rec["cost_logged_usd"] for rec in records)
        quality = sum(routed if id(rec) in adopted else logged
                      for _, _, rec, logged, routed in candidates) / len(records)
        rows.append({"policy": policy, "adopt_frac": frac, "cost_usd": round(cost, 4),
                     "expected_success": round(quality, 3)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smart", nargs="?", default="results/routes_smart.jsonl")
    parser.add_argument("--baseline", default="results/routes.jsonl")
    parser.add_argument("--outcomes", default="results/outcomes.json")
    parser.add_argument("--features", default="results/call_features.csv")
    parser.add_argument("--output", default="results/frontier_2.csv")
    args = parser.parse_args()
    configure_estimator(args.outcomes, args.features)
    smart, baseline = curve(load(args.smart), "smart"), curve(load(args.baseline), "baseline")
    rows = baseline + smart
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    dominates = 0
    print("adoption  baseline(cost,quality)  smart(cost,quality)  smart_up_left")
    for base, new in zip(baseline, smart):
        up_left = new["cost_usd"] <= base["cost_usd"] and new["expected_success"] >= base["expected_success"]
        dominates += up_left
        print(f" {new['adopt_frac']:4.0%}     ${base['cost_usd']:.4f},{base['expected_success']:.3f}       "
              f"${new['cost_usd']:.4f},{new['expected_success']:.3f}       {up_left}")
    print(f"smart weakly up-and-left at {dominates}/{len(smart)} equal-adoption points")
    print(f"wrote {output}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, values, color in (("baseline", baseline, "#999999"), ("smart", smart, "#6748FD")):
            plt.plot([r["cost_usd"] for r in values], [r["expected_success"] for r in values],
                     "o-", label=name, color=color)
        plt.xlabel("cost (USD, estimated input tokens, cache-aware)")
        plt.ylabel("matched expected success")
        plt.title("Smart vs baseline frontier"); plt.legend(); plt.tight_layout()
        png = output.with_suffix(".png"); plt.savefig(png, dpi=150); print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed - skipped PNG")


if __name__ == "__main__":
    main()
