#!/usr/bin/env python3
"""Extract call-level routing features and an observable-action complexity proxy.

The export contains requests, not responses.  For every non-final call, the model's
response is recovered from the items appended to the next request.  Final-call
outcomes are deliberately left unknown and excluded from proxy correlations.

Usage:
    python scripts/feature_analyzer.py export/
    python scripts/feature_analyzer.py export/ --output results/call_features.csv

Only the Python standard library is required.  The CSV is directly loadable with
``pandas.read_csv`` if a DataFrame is desired.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# The bundled isolated Windows runtime does not automatically add the script's
# directory to sys.path; make the sibling loader import reliable everywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_trajectories import est_tokens, group_trajectories, iter_requests


OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}
CALL_TYPES = {"function_call", "custom_tool_call"}
SHELL_NAMES = {"run_shell", "shell", "exec", "exec_command", "bash"}


def item_type(item: dict[str, Any]) -> str:
    return item.get("type", "message")


def message_text(item: dict[str, Any]) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(part.get("text", "")) for part in content
        if isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}
    )


def stable_hash(text: str) -> str:
    return hashlib.sha1(text[:4000].encode("utf-8")).hexdigest()[:12]


def opening_text(req: dict[str, Any]) -> str:
    bits = []
    for item in req["input"]:
        if item.get("role") in {"system", "user"}:
            bits.append(message_text(item))
        if item.get("role") == "user":
            break
    return "\n".join(bits)


def task_category(text: str) -> str:
    """A deliberately interpretable, overlapping-keyword task taxonomy."""
    text = text.lower()
    groups = {
        "coding": r"\b(code|bug|implement|repository|script|python|javascript|api|sql|test|deploy)\b",
        "data_analysis": r"\b(analy[sz]|dataset|spreadsheet|csv|metric|chart|statistics|forecast)\b",
        "research": r"\b(research|find|search|source|compare|investigate|report)\b",
        "communication": r"\b(email|message|write|rewrite|summari[sz]|translate|document|presentation)\b",
        "planning": r"\b(plan|schedule|strategy|roadmap|organize|book|travel)\b",
        "visual": r"\b(image|photo|design|diagram|visual|slide|pdf)\b",
    }
    hits = [(name, len(re.findall(pattern, text))) for name, pattern in groups.items()]
    name, count = max(hits, key=lambda pair: pair[1])
    return name if count else "general"


def output_text(item: dict[str, Any]) -> str:
    value = item.get("output", "")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def recent_tool_output_tokens(items: list[dict[str, Any]], recent: int = 3) -> int:
    outputs = [output_text(item) for item in items if item_type(item) in OUTPUT_TYPES]
    return sum(est_tokens(text) for text in outputs[-recent:])


def appended_items(current: dict[str, Any], following: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover new history items, tolerating a malformed/non-prefix reconstruction."""
    before, after = current["input"], following["input"]
    common = 0
    for left, right in zip(before, after):
        if left != right:
            break
        common += 1
    return after[common:]


def parse_arguments(item: dict[str, Any]) -> tuple[str, str]:
    raw = item.get("arguments", item.get("input", ""))
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    command = ""
    try:
        args = json.loads(raw)
        if isinstance(args, dict):
            command = str(args.get("cmd", args.get("command", args.get("patch", ""))))
    except (ValueError, TypeError):
        pass
    return raw, command


def action_proxy(new_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe model-produced items, excluding environment/tool results.

    ``simple`` means a short direct answer, or one short low-branching tool call.
    It is an action-shape proxy only; it is not evidence another model would succeed.
    """
    generated = [item for item in new_items if item_type(item) not in OUTPUT_TYPES]
    calls = [item for item in generated if item_type(item) in CALL_TYPES]
    assistants = [item for item in generated if item.get("role") == "assistant"]
    reasoning = [item for item in generated if item_type(item) == "reasoning"]
    arg_chars = command_chars = shell_ops = multiline = 0
    tool_names = []
    for item in calls:
        tool_names.append(str(item.get("name", item.get("type", "unknown"))))
        raw, command = parse_arguments(item)
        arg_chars += len(raw)
        command_chars += len(command)
        shell_ops += len(re.findall(r"(?:&&|\|\||[|;])", command))
        multiline += int("\n" in command or "\r" in command)
    assistant_tokens = sum(est_tokens(message_text(item)) for item in assistants)
    action_tokens = sum(est_tokens(item) for item in generated)
    direct = bool(assistants) and not calls
    simple = (
        (direct and assistant_tokens <= 160)
        or (len(calls) == 1 and arg_chars <= 240 and shell_ops <= 1 and multiline == 0)
    )
    # Continuous proxy is useful beyond the intentionally conservative binary label.
    score = (
        math.log1p(action_tokens) + 1.4 * len(calls) + 0.6 * shell_ops
        + 0.8 * multiline + 0.5 * max(0, len(calls) - 1)
    )
    kind = "direct_answer" if direct else ("tool_call" if calls else "other")
    return {
        "proxy_simple": int(simple),
        "proxy_score": round(score, 6),
        "action_kind": kind,
        "action_tokens_est": action_tokens,
        "action_tool_calls": len(calls),
        "action_tool_names": "|".join(tool_names),
        "action_argument_chars": arg_chars,
        "action_command_chars": command_chars,
        "action_shell_operators": shell_ops,
        "action_multiline_commands": multiline,
        "action_reasoning_items": len(reasoning),
    }


def extract_rows(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for trajectory, calls in groups.items():
        initial = opening_text(calls[0])
        system = next((message_text(i) for i in calls[0]["input"] if i.get("role") == "system"), "")
        category = task_category(initial)
        for index, call in enumerate(calls):
            total = len(calls)
            if index == 0:
                phase = "opening"
            elif index == total - 1:
                phase = "final_verification"
            else:
                phase = "middle"
            items = call["input"]
            tools = call.get("tools", [])
            row = {
                "trajectory": trajectory,
                "model": call["model"],
                "call_index": index + 1,
                "trajectory_calls": total,
                "position_fraction": round(index / max(1, total - 1), 6),
                "phase": phase,
                "prompt_tokens_est": est_tokens(items),
                "tool_count": len(tools),
                "tool_schema_chars": len(json.dumps(tools, ensure_ascii=False)),
                "tool_schema_tokens_est": est_tokens(tools),
                "recent_tool_output_tokens_est": recent_tool_output_tokens(items),
                "has_input_image": int(any(
                    isinstance(part, dict) and part.get("type") == "input_image"
                    for item in items for part in (item.get("content") if isinstance(item.get("content"), list) else [])
                )),
                "has_reasoning": int(any(item_type(item) == "reasoning" for item in items)),
                "task_hash": stable_hash(initial),
                "system_hash": stable_hash(system),
                "task_category": category,
                "outcome_observed": int(index + 1 < total),
            }
            if index + 1 < total:
                row.update(action_proxy(appended_items(call, calls[index + 1])))
            else:
                row.update({key: "" for key in (
                    "proxy_simple", "proxy_score", "action_kind", "action_tokens_est",
                    "action_tool_calls", "action_tool_names", "action_argument_chars",
                    "action_command_chars", "action_shell_operators",
                    "action_multiline_commands", "action_reasoning_items",
                )})
            rows.append(row)
    return rows


def assign_proxy_clusters(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Split observed action scores with deterministic one-dimensional k-means.

    Dataset-relative clustering avoids pretending that an arbitrary command-length
    threshold is universal.  The lower-complexity cluster is named ``simple``.
    """
    observed = [row for row in rows if row["outcome_observed"]]
    values = [float(row["proxy_score"]) for row in observed]
    if not values:
        return 0.0, 0.0
    low, high = min(values), max(values)
    if low == high:
        for row in observed:
            row["proxy_simple"] = 1
        return low, high
    for _ in range(100):
        left, right = [], []
        for value in values:
            (left if abs(value - low) <= abs(value - high) else right).append(value)
        new_low = statistics.fmean(left) if left else low
        new_high = statistics.fmean(right) if right else high
        if abs(new_low - low) + abs(new_high - high) < 1e-9:
            break
        low, high = new_low, new_high
    low, high = sorted((low, high))
    boundary = (low + high) / 2
    for row in observed:
        row["proxy_simple"] = int(float(row["proxy_score"]) <= boundary)
    return low, high


def pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)]
    if len(pairs) < 2:
        return 0.0
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return numerator / denominator if denominator else 0.0


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def print_report(rows: list[dict[str, Any]], output: Path, cluster_centers: tuple[float, float]) -> None:
    observed = [row for row in rows if row["outcome_observed"]]
    simple_rate = statistics.fmean(row["proxy_simple"] for row in observed) if observed else 0.0
    numeric = [
        "prompt_tokens_est", "call_index", "trajectory_calls", "position_fraction",
        "tool_count", "tool_schema_tokens_est", "recent_tool_output_tokens_est",
        "has_input_image", "has_reasoning",
    ]
    correlations = sorted(
        ((name, pearson([row[name] for row in observed], [row["proxy_simple"] for row in observed])) for name in numeric),
        key=lambda pair: abs(pair[1]), reverse=True,
    )
    print("\n=== Routing feature analysis ===")
    print(f"calls={len(rows):,}  trajectories={len({r['trajectory'] for r in rows}):,}  "
          f"proxy-observed={len(observed):,}  final/outcome-unknown={len(rows)-len(observed):,}")
    print(f"simple-action proxy rate: {pct(simple_rate)}")
    print(f"action-score cluster centers: simple={cluster_centers[0]:.3f}, heavy={cluster_centers[1]:.3f}")
    print("\nNumeric association with simple-action proxy (Pearson / point-biserial):")
    for name, correlation in correlations:
        direction = "more simple" if correlation > 0 else "less simple"
        print(f"  {name:32s} {correlation:+.3f}  ({direction})")
    print("\nSimple-action rates by nameable structure (n >= 3):")
    for field in ("phase", "task_category", "action_kind", "model"):
        buckets: dict[str, list[int]] = defaultdict(list)
        for row in observed:
            buckets[str(row[field])].append(row["proxy_simple"])
        rendered = [
            f"{value}={pct(statistics.fmean(labels))} (n={len(labels)})"
            for value, labels in sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0]))
            if len(labels) >= 3
        ]
        print(f"  {field}: " + "; ".join(rendered))
    print("\nInterpretation guardrail: this proxy measures the logged model's next action shape,")
    print("not correctness or the counterfactual quality of a cheaper model. Correlations are")
    print("descriptive and confounded by logged-model/task selection; final calls are excluded.")
    print(f"\nWrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export", help="directory containing export/*.jsonl")
    parser.add_argument("--output", default="results/call_features.csv", help="feature CSV path")
    args = parser.parse_args()
    groups = group_trajectories(req for _, _, req in iter_requests(args.export))
    rows = extract_rows(groups)
    cluster_centers = assign_proxy_clusters(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_report(rows, output, cluster_centers)


if __name__ == "__main__":
    main()
