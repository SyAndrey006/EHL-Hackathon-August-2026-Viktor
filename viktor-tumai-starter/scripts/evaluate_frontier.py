#!/usr/bin/env python3
"""Evaluate the starter router's cache-aware cost and heuristic outcome baseline.

Usage: python scripts/evaluate_frontier.py [export_directory]

Costs include estimated input tokens only: the export contains neither measured
usage nor final outputs.  The heuristic outcome is observational and does not
estimate the quality impact of counterfactual routing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from baseline_router import route_trajectory
from cost_model import load_pricing, logged_route, trajectory_cost
from load_trajectories import group_trajectories, iter_requests
from outcomes import evaluate_trajectory_outcome


def evaluate(export_dir: Path) -> tuple[int, float, float, float]:
    """Return task count, logged cost, heuristic success rate, and routed cost."""
    requests = (request for _, _, request in iter_requests(export_dir))
    trajectories = group_trajectories(requests)
    pricing = load_pricing()

    baseline_cost = 0.0
    routed_cost = 0.0
    heuristic_successes = 0

    for calls in trajectories.values():
        outcome = evaluate_trajectory_outcome(pd.DataFrame.from_records(calls))
        # Per the baseline definition, every task not positively flagged by the
        # conservative failure heuristic is counted as a heuristic success.
        heuristic_successes += int(not outcome["heuristic_failure"])

        original_route = logged_route(calls)
        new_route = route_trajectory(calls)
        original_trajectory_cost, _ = trajectory_cost(
            calls, original_route, pricing
        )
        routed_trajectory_cost, _ = trajectory_cost(calls, new_route, pricing)
        baseline_cost += original_trajectory_cost
        routed_cost += routed_trajectory_cost

    total_tasks = len(trajectories)
    success_rate = heuristic_successes / total_tasks if total_tasks else 0.0
    return total_tasks, baseline_cost, success_rate, routed_cost


def _print_summary(
    total_tasks: int,
    baseline_cost: float,
    success_rate: float,
    routed_cost: float,
) -> None:
    """Print a compact terminal report."""
    width = 64
    print("=" * width)
    print("COST-QUALITY FRONTIER BASELINE".center(width))
    print("=" * width)
    print(f"{'Total Tasks':<36}{total_tasks:>22,}")
    print(f"{'Baseline Cost':<36}{baseline_cost:>21,.4f} USD")
    print(f"{'Baseline Heuristic Success Rate':<36}{success_rate:>21.2%}")
    print(f"{'New Routed Cost':<36}{routed_cost:>21,.4f} USD")
    print("-" * width)
    print("Costs are cache-aware input-side estimates; output cost excluded.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline heuristic quality and cache-aware costs."
    )
    parser.add_argument(
        "export_dir",
        nargs="?",
        type=Path,
        default=Path("export"),
        help="directory containing trajectories_v1_*.jsonl (default: export)",
    )
    args = parser.parse_args(argv)
    _print_summary(*evaluate(args.export_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
