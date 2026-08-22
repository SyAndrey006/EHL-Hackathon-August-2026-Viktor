"""Outcome proxies for reconstructed agent trajectories.

The request export has no final model output or ground-truth task label.  This
module therefore exposes a deliberately conservative error-based proxy and a
clean transcript suitable for a future off-policy LLM judge.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Mapping

import pandas as pd


_ERROR_RE = re.compile(
    r"(?:\bexception\b|\btraceback\b|\berror\b|\bfailed\b|\bfailure\b|"
    r"\btime[- ]?out\b|\btimed out\b)",
    flags=re.IGNORECASE,
)
_NEGATED_ERROR_RE = re.compile(
    r"\b(?:no|without|zero)\s+(?:errors?|exceptions?|failures?)\b|"
    r"\b(?:errors?|failures?)\s*[:=]\s*(?:false|none|0)\b",
    flags=re.IGNORECASE,
)
_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})
_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})


def _is_missing(value: object) -> bool:
    """Return whether a scalar value is a pandas-style missing value."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _as_items(value: object) -> list[Mapping[str, Any]]:
    """Normalize a Responses-format input value, rejecting malformed items."""
    if _is_missing(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _trajectory_items(trajectory_calls: pd.DataFrame) -> list[Mapping[str, Any]]:
    """Recover the most complete cumulative history from trajectory rows."""
    if not isinstance(trajectory_calls, pd.DataFrame):
        raise TypeError("trajectory_calls must be a pandas DataFrame")
    if trajectory_calls.empty or "input" not in trajectory_calls.columns:
        return []

    histories = [_as_items(value) for value in trajectory_calls["input"].tolist()]
    # Reconstructed calls should be prefix histories.  The longest one contains
    # every observable action, including outputs revealed to subsequent calls.
    return max(histories, key=len, default=[])


def _stringify(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return _stringify(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            part_type = str(part.get("type", ""))
            if part_type in {"input_image", "output_image", "image"}:
                parts.append("[image]")
            else:
                text = part.get("text", part.get("content", ""))
                if text != "":
                    parts.append(_stringify(text))
    return "\n".join(part for part in parts if part)


def _output_text(item: Mapping[str, Any]) -> str:
    for key in ("output", "content", "result", "text"):
        if key in item:
            return _stringify(item[key])
    return ""


def _contains_error(text: str) -> bool:
    cleaned = _NEGATED_ERROR_RE.sub("", text)
    return bool(_ERROR_RE.search(cleaned))


def _call_id(item: Mapping[str, Any]) -> str:
    return str(item.get("call_id", item.get("id", "")))


def evaluate_trajectory_outcome(trajectory_calls: pd.DataFrame) -> dict[str, Any]:
    """Estimate task success from observable intermediate tool outputs.

    A trajectory is marked as a heuristic failure only when its observable tool
    output sequence ends in a cluster of errors (at least two of the last three
    outputs, including the final output), or the same named tool has at least
    two consecutive failed calls.  A lack of observable outputs is reported as
    ``unknown`` rather than optimistically treated as success.
    """
    items = _trajectory_items(trajectory_calls)
    calls_by_id: dict[str, str] = {}
    output_events: list[tuple[str, bool]] = []

    for item in items:
        item_type = str(item.get("type", ""))
        if item_type in _CALL_TYPES:
            calls_by_id[_call_id(item)] = str(
                item.get("name", item.get("tool_name", "unknown_tool"))
            )
        elif item_type in _OUTPUT_TYPES:
            name = calls_by_id.get(_call_id(item), "unknown_tool")
            output_events.append((name, _contains_error(_output_text(item))))

    error_count = sum(is_error for _, is_error in output_events)
    tail = output_events[-3:]
    terminal_error_cluster = (
        len(tail) >= 2
        and tail[-1][1]
        and sum(is_error for _, is_error in tail) >= 2
    )

    repeated_failed_calls = False
    for previous, current in zip(output_events, output_events[1:]):
        if previous[0] == current[0] and previous[1] and current[1]:
            repeated_failed_calls = True
            break

    reasons: list[str] = []
    if terminal_error_cluster:
        reasons.append("terminal_tool_error_cluster")
    if repeated_failed_calls:
        reasons.append("repeated_failed_tool_call")
    heuristic_failure = bool(reasons)
    if not output_events:
        label = "unknown"
    else:
        label = "failure" if heuristic_failure else "success"

    failed_tools = Counter(name for name, failed in output_events if failed)
    return {
        "outcome_label": label,
        "heuristic_failure": heuristic_failure,
        "heuristic_success": label == "success",
        "observable_tool_outputs": len(output_events),
        "error_output_count": error_count,
        "error_rate": error_count / len(output_events) if output_events else None,
        "terminal_error_cluster": terminal_error_cluster,
        "repeated_failed_calls": repeated_failed_calls,
        "failed_tools": dict(sorted(failed_tools.items())),
        "failure_reasons": reasons,
    }


def _format_call(item: Mapping[str, Any]) -> str:
    name = str(item.get("name", item.get("tool_name", "unknown_tool")))
    arguments = item.get("arguments", item.get("input", {}))
    return f"TOOL CALL [{name}] (id={_call_id(item) or 'unknown'}): {_stringify(arguments)}"


def format_transcript_for_judge(trajectory_calls: pd.DataFrame) -> str:
    """Format all observable actions as a chronological, judge-ready transcript."""
    items = _trajectory_items(trajectory_calls)
    lines: list[str] = []
    calls_by_id: dict[str, str] = {}

    for item in items:
        item_type = str(item.get("type", ""))
        role = item.get("role")
        if role is not None or item_type == "message":
            text = _content_text(item.get("content", item.get("text", ""))).strip()
            if text:
                lines.append(f"{str(role or 'message').upper()}: {text}")
        elif item_type in _CALL_TYPES:
            calls_by_id[_call_id(item)] = str(
                item.get("name", item.get("tool_name", "unknown_tool"))
            )
            lines.append(_format_call(item))
        elif item_type in _OUTPUT_TYPES:
            call_id = _call_id(item)
            name = calls_by_id.get(call_id, "unknown_tool")
            lines.append(
                f"TOOL OUTPUT [{name}] (id={call_id or 'unknown'}): "
                f"{_output_text(item)}"
            )
        elif item_type == "reasoning":
            summary = _content_text(item.get("summary", "")).strip()
            if summary:
                lines.append(f"REASONING SUMMARY: {summary}")

    return "\n\n".join(lines)


__all__ = ["evaluate_trajectory_outcome", "format_transcript_for_judge"]
