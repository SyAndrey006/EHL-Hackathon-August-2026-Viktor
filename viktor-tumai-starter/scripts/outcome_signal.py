#!/usr/bin/env python3
"""Transparent structural outcome proxy for reconstructed trajectories.

There is no final output or human quality label in this export.  This module
therefore estimates *observed execution health*, not correctness: a score near
one means the recoverable history looks like a clean, completed execution;
zero means it contains several visible signs of trouble.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from load_trajectories import first_user_text, group_trajectories, iter_requests


ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|timeout|timed out|"
    r"permission denied|not found|invalid|unauthorized|cannot)\b", re.I
)
DONE_RE = re.compile(
    r"\b(done|completed|complete|finished|resolved|verified|successfully|"
    r"all set|implemented|fixed)\b", re.I
)


def _text(value):
    """Flatten an output/content value into searchable text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _history_items(calls):
    """The final request contains the longest recoverable history."""
    return calls[-1].get("input", []) if calls else []


def _normal_args(arguments):
    """Normalize superficial differences before comparing retry arguments."""
    text = _text(arguments).lower()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"\s+", " ", text).strip()


def _similar_args(left, right):
    """Require high text similarity *and* high argument-token overlap.

    The token check prevents sequential calls such as ``step-0`` and ``step-1``
    from being treated as retries merely because they share long boilerplate.
    A retry should repeat its relevant target/path/command tokens as well.
    """
    if left == right:
        return True
    char_similarity = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(re.findall(r"[\w./:-]+", left))
    right_tokens = set(re.findall(r"[\w./:-]+", right))
    token_union = left_tokens | right_tokens
    token_overlap = len(left_tokens & right_tokens) / len(token_union) if token_union else 1.0
    return char_similarity >= 0.82 and token_overlap >= 0.80


def outcome_features(calls, expected_calls=None):
    """Return the named, auditable ingredients of the proxy."""
    if not calls:
        return {"n_calls": 0, "error_outputs": 0, "retry_pairs": 0,
                "call_outlier": 0.0, "has_done_message": False,
                "abandoned_early": True, "expected_calls": 1.0}

    items = _history_items(calls)
    outputs = [i for i in items if i.get("type") in
               ("function_call_output", "custom_tool_call_output")]
    error_outputs = sum(bool(ERROR_RE.search(_text(i.get("output", "")))) for i in outputs)

    recent = []
    retry_pairs = 0
    for item in items:
        if item.get("type") not in ("function_call", "custom_tool_call"):
            continue
        name = item.get("name", "")
        args = _normal_args(item.get("arguments", item.get("input", "")))
        for old_name, old_args in recent[-4:]:
            if name == old_name and _similar_args(args, old_args):
                retry_pairs += 1
                break
        recent.append((name, args))

    assistant_text = " ".join(
        _text(i.get("content", "")) for i in items
        if i.get("type") == "message" and i.get("role") == "assistant"
    )
    has_done = bool(DONE_RE.search(assistant_text))

    if expected_calls is None:
        prompt_chars = len(first_user_text(calls[0]))
        expected_calls = 3.0 + min(4.0, prompt_chars / 1500.0)
    expected_calls = max(float(expected_calls), 1.0)
    call_outlier = max(0.0, len(calls) / expected_calls - 1.0)
    abandoned = len(calls) >= 3 and bool(outputs) and not has_done
    return {"n_calls": len(calls), "error_outputs": error_outputs,
            "retry_pairs": retry_pairs, "call_outlier": call_outlier,
            "has_done_message": has_done, "abandoned_early": abandoned,
            "expected_calls": expected_calls}


def estimate_outcome(calls, expected_calls=None):
    """Estimate observed execution health on [0, 1]; higher is healthier."""
    f = outcome_features(calls, expected_calls)
    error_penalty = min(0.35, 0.12 * f["error_outputs"])
    retry_penalty = min(0.25, 0.10 * f["retry_pairs"])
    outlier_penalty = min(0.20, 0.12 * f["call_outlier"])
    abandoned_penalty = 0.20 if f["abandoned_early"] else 0.0
    return round(max(0.0, 1.0 - error_penalty - retry_penalty - outlier_penalty - abandoned_penalty), 4)


def cohort_key(calls):
    """Coarse task-shape bucket: tool set + first-user prompt length quartile."""
    tools = tuple(sorted(t.get("name", "") for t in calls[0].get("tools", []) if t.get("name"))) if calls else ()
    chars = len(first_user_text(calls[0])) if calls else 0
    return tools, min(3, chars // 1000)


def cohort_expected_calls(groups):
    """Median call count by coarse task shape; sparse cohorts fall back globally."""
    counts = defaultdict(list)
    all_counts = []
    for calls in groups.values():
        counts[cohort_key(calls)].append(len(calls))
        all_counts.append(len(calls))
    global_median = sorted(all_counts)[len(all_counts) // 2] if all_counts else 1
    return {key: (sorted(values)[len(values) // 2] if len(values) >= 3 else global_median)
            for key, values in counts.items()}


def main():
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    expected = cohort_expected_calls(groups)
    scores, totals = [], Counter()
    for calls in groups.values():
        f = outcome_features(calls, expected[cohort_key(calls)])
        scores.append(estimate_outcome(calls, f["expected_calls"]))
        totals.update({"error_outputs": f["error_outputs"], "retry_pairs": f["retry_pairs"],
                       "potentially_abandoned": int(f["abandoned_early"])})
    print(f"trajectories={len(scores)}  mean outcome={sum(scores)/len(scores):.3f}  "
          f"min/median/max={min(scores):.3f}/{sorted(scores)[len(scores)//2]:.3f}/{max(scores):.3f}")
    print("visible signals:", dict(totals))
    print("NOTE: outcome is an execution-health proxy, not observed correctness; see source for failure modes.")


if __name__ == "__main__":
    main()
