#!/usr/bin/env python3
"""Plot cost/health adoption frontiers from routed OPE results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATCH_FIELDS = ("first_user_chars", "first_input_tokens", "tool_count", "tool_set")
OUTCOME_FIELDS = ("execution_health", "health", "outcome", "reward", "success")
SWEEP_POINTS = 101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adoption frontiers using matched exported OPE outcomes."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more route .jsonl files followed by the export directory.",
    )
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("provide at least one route .jsonl file and the export directory")
    args.route_files, args.export_dir = args.inputs[:-1], args.inputs[-1]
    if not args.export_dir.is_dir():
        parser.error(f"export directory does not exist: {args.export_dir}")
    for path in args.route_files:
        if path.suffix.lower() != ".jsonl" or not path.is_file():
            parser.error(f"route input must be an existing .jsonl file: {path}")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def exported_rows(export_dir: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(export_dir.rglob("*.jsonl")):
        yield from read_jsonl(path)
    for path in sorted(export_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            yield from (row for row in value if isinstance(row, dict))
        elif isinstance(value, dict):
            for key in ("trajectories", "rows", "data", "records"):
                if isinstance(value.get(key), list):
                    yield from (row for row in value[key] if isinstance(row, dict))
                    break
            else:
                yield value


def first_present(row: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize_tool_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            value = decoded if isinstance(decoded, (list, tuple, set)) else value.split(",")
        except json.JSONDecodeError:
            value = value.split(",")
    if isinstance(value, dict):
        value = list(value)
    if not isinstance(value, (list, tuple, set, frozenset)):
        value = [value]
    return frozenset(str(item).strip().lower() for item in value if str(item).strip())


def model_of(row: dict[str, Any], routed: bool = False) -> str:
    names = (
        ("routed_model", "route_model", "selected_model", "target_model", "model")
        if routed else ("logged_model", "source_model", "model_name", "model")
    )
    return str(first_present(row, names, "")).strip()


def outcome_of(row: dict[str, Any]) -> float | None:
    value = first_present(row, OUTCOME_FIELDS)
    if value is None and isinstance(row.get("metrics"), dict):
        value = first_present(row["metrics"], OUTCOME_FIELDS)
    return None if value is None else as_float(value)


def distance(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    numeric = 0.0
    for field in MATCH_FIELDS[:3]:
        left, right = as_float(query.get(field)), as_float(candidate.get(field))
        numeric += abs(left - right) / max(left, right, 1.0)
    left_set = normalize_tool_set(query.get("tool_set"))
    right_set = normalize_tool_set(candidate.get("tool_set"))
    union = left_set | right_set
    set_distance = 0.0 if not union else 1.0 - len(left_set & right_set) / len(union)
    return numeric + set_distance


class Matcher:
    def __init__(self, trajectories: Iterable[dict[str, Any]]) -> None:
        self.by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in trajectories:
            if outcome_of(row) is not None:
                self.by_model[model_of(row)].append(row)
        if not self.by_model:
            raise ValueError("no exported trajectories with health outcomes found")

    def outcome(self, query: dict[str, Any], model: str) -> float:
        candidates = self.by_model.get(model)
        if not candidates:
            raise ValueError(f"no exported trajectories for model {model!r}")
        value = outcome_of(min(candidates, key=lambda row: distance(query, row)))
        assert value is not None
        return value


def prepare(rows: list[dict[str, Any]], matcher: Matcher) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row in rows:
        logged_model, routed_model = model_of(row), model_of(row, True)
        if not logged_model or not routed_model:
            raise ValueError("route row is missing logged_model or routed_model")
        result.append({
            "cost_logged": as_float(row.get("cost_logged_usd")),
            "cost_routed": as_float(row.get("cost_routed_usd")),
            "n_calls": max(0.0, as_float(row.get("n_calls"), 1.0)),
            "logged_health": matcher.outcome(row, logged_model),
            "routed_health": matcher.outcome(row, routed_model),
        })
    return sorted(result, key=lambda row: row["cost_logged"])


def point(rows: list[dict[str, float]], adopted_count: int) -> tuple[float, float]:
    cost = weighted_health = total_calls = 0.0
    for index, row in enumerate(rows):
        adopted = index < adopted_count
        cost += row["cost_routed"] if adopted else row["cost_logged"]
        health = row["routed_health"] if adopted else row["logged_health"]
        weighted_health += health * row["n_calls"]
        total_calls += row["n_calls"]
    return cost, weighted_health / total_calls if total_calls else 0.0


def sweep(rows: list[dict[str, float]]) -> list[tuple[float, float, float]]:
    points = []
    for step in range(SWEEP_POINTS):
        fraction = step / (SWEEP_POINTS - 1)
        count = min(len(rows), int(round(fraction * len(rows))))
        cost, health = point(rows, count)
        points.append((fraction, cost, health))
    return points


def extreme(rows: list[dict[str, float]], cheapest: bool) -> tuple[float, float]:
    cost = weighted_health = total_calls = 0.0
    for row in rows:
        route_is_cheaper = row["cost_routed"] <= row["cost_logged"]
        routed = route_is_cheaper if cheapest else not route_is_cheaper
        cost += row["cost_routed"] if routed else row["cost_logged"]
        health = row["routed_health"] if routed else row["logged_health"]
        weighted_health += health * row["n_calls"]
        total_calls += row["n_calls"]
    return cost, weighted_health / total_calls if total_calls else 0.0


def main() -> None:
    args = parse_args()
    matcher = Matcher(exported_rows(args.export_dir))
    prepared: dict[str, list[dict[str, float]]] = {}
    series: dict[str, list[tuple[float, float, float]]] = {}
    for path in args.route_files:
        label = path.stem.replace("_routes", "").replace("_", " ").title()
        prepared[label] = prepare(read_jsonl(path), matcher)
        series[label] = sweep(prepared[label])

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = next(iter(prepared.values()))
    references = {
        "always-cheapest": extreme(reference, True),
        "always-most-expensive": extreme(reference, False),
    }
    with (output_dir / "frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("series", "adoption_fraction", "cost_usd", "execution_health"))
        for label, points in series.items():
            writer.writerows((label, fraction, cost, health) for fraction, cost, health in points)
        for label, (cost, health) in references.items():
            writer.writerow((label, "", cost, health))

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, points in series.items():
        ax.plot([p[1] for p in points], [p[2] for p in points], label=label, linewidth=2)
    for (label, (cost, health)), marker in zip(references.items(), ("v", "^")):
        ax.scatter([cost], [health], label=label, marker=marker, s=90, zorder=5)
    ax.set(xlabel="Total cost (USD)", ylabel="Weighted average execution health",
           title="Routing cost-health frontier")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "frontier.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
