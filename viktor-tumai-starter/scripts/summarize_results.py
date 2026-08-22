#!/usr/bin/env python3
"""Print the short, honest result summary needed for the pitch/weakness slide.

Usage:
  python scripts/summarize_results.py results/baseline_routes.jsonl \
      results/learned_routes.jsonl [results/frontier.csv]
"""
import csv
import json
import sys
from pathlib import Path


def route_cost(path):
    records = [json.loads(line) for line in open(path) if line.strip()]
    return sum(r["cost_routed_usd"] for r in records), len(records)


def endpoint(rows, policy):
    candidates = [r for r in rows if r["policy"] == policy]
    return max(candidates, key=lambda r: float(r["adopt_frac_requested"])) if candidates else None


def main():
    baseline = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/baseline_routes.jsonl")
    learned = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/learned_routes.jsonl")
    frontier = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/frontier.csv")
    baseline_cost, n_baseline = route_cost(baseline)
    learned_cost, n_learned = route_cost(learned)
    if n_baseline != n_learned:
        raise SystemExit("route files cover different trajectory counts; do not compare them")
    rows = list(csv.DictReader(open(frontier)))
    base_end, learned_end = endpoint(rows, "baseline"), endpoint(rows, "learned")
    print("=== Router evaluation summary ===")
    print(f"trajectories compared: {n_baseline}")
    print(f"input-side routed cost — baseline: ${baseline_cost:,.4f}; learned: ${learned_cost:,.4f}")
    print(f"learned vs baseline cost delta: ${learned_cost - baseline_cost:,.4f} "
          f"({learned_cost / baseline_cost - 1:+.1%})")
    if base_end and learned_end:
        print("matched-OPE execution health at full requested adoption — "
              f"baseline: {base_end['estimated_execution_health']}; "
              f"learned: {learned_end['estimated_execution_health']}")
        print("common support at full adoption — "
              f"baseline adopted/unsupported: {base_end['adopted_supported']}/{base_end['unsupported_proposals']}; "
              f"learned adopted/unsupported: {learned_end['adopted_supported']}/{learned_end['unsupported_proposals']}")
    print("Outcome signal: recoverable execution health (tool errors, close retries, excess calls, "
          "and no visible completion), NOT final-answer correctness or user satisfaction.")
    print("Known failure modes: final response/output cost are absent; successful recovery can look "
          "like failure; long legitimate tasks can look inefficient; keyword matching can miss errors.")
    print("OPE assumption: matched traces are comparable after prompt/input size and tool features. "
          "It breaks under unobserved task difficulty, weak common support, post-treatment call-count "
          "matching, reused/asymmetric neighbours, or proxy-label error.")
    print("All dollar figures are cache-aware INPUT-SIDE estimates using chars/4 token estimates; "
          "they exclude unknown output usage/cost.")


if __name__ == "__main__":
    main()
