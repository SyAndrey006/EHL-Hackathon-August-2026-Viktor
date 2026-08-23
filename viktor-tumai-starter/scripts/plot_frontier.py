#!/usr/bin/env python3
"""Matched expected-success vs cache-aware input-cost frontier.

Usage: python scripts/plot_frontier.py results/routes.jsonl
Writes results/frontier.csv (+ frontier.png if matplotlib is installed).
"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome_evaluator import configure_estimator, estimate_route_quality


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/routes.jsonl"
    recs = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    results_dir = Path(src).parent
    configure_estimator(results_dir / "outcomes.json", results_dir / "call_features.csv")
    by_size = sorted(recs, key=lambda rec: rec["cost_logged_usd"])
    rows = []
    for frac in [i / 10 for i in range(11)]:
        adopt = {id(rec) for rec in by_size[:int(round(frac * len(recs)))]}
        cost = sum(rec["cost_routed_usd"] if id(rec) in adopt else rec["cost_logged_usd"] for rec in recs)
        expected = sum(
            estimate_route_quality(
                rec["trajectory"],
                rec["route"] if id(rec) in adopt else [rec["logged_model"]] * rec["n_calls"],
            ) for rec in recs
        )
        rows.append({"adopt_frac": frac, "cost_usd": round(cost, 4),
                     "expected_success": round(expected / len(recs), 3)})
    out = results_dir / "frontier.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out}")
    for row in rows:
        print(row)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot([r["cost_usd"] for r in rows], [r["expected_success"] for r in rows],
                 "o-", color="#6748FD")
        plt.xlabel("cost (USD, estimated tokens, cache-aware)")
        plt.ylabel("matched expected success")
        plt.title("Cost-quality frontier")
        plt.tight_layout()
        png = results_dir / "frontier.png"; plt.savefig(png, dpi=150)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed - skipped PNG (CSV is enough)")


if __name__ == "__main__":
    main()
