#!/usr/bin/env python3
"""Transparent structural outcome proxy for reconstructed trajectories.

There is no final output or human quality label in this export.  This module
therefore estimates *observed execution health*, not correctness: a score near
one means the recoverable history looks like a clean, completed execution;
zero means it contains several visible signs of trouble.

Usage: python scripts/outcome_signal.py export/
Import: estimate_outcome(calls) -> float in [0, 1]

The optional ``expected_calls`` argument lets an evaluator supply the median
call count for a cohort with the same tool set and similarly sized first user
prompt.  Without it, a deliberately conservative complexity-based expectation
is used, so the advertised one-argument API remains rerunnable.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from load_trajectories import first_user_text, group_trajectories, iter_requests


# These strings are intentionally a small, inspectable list.  A match in a
# function_call_output is a sign of an unsuccessful tool interaction, not proof
# that the task ultimately failed (the agent may recover on its next call).
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
    text = re.sub(r"\d+", "#", text)       # generated ids/line numbers
    return re.sub(r"\s+", " ", text).strip()


def _similar_args(left, right):
    """Require both high character similarity and mostly identical argument tokens.

    The token check prevents a sequence such as ``step-0``, ``step-1`` from
    looking like a retry just because both commands contain a long shared log
    string. A real retry normally repeats its target/path/command tokens too.
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
    """Return the named, auditable ingredients of the proxy.

    ``calls`` must be one trajectory, ordered or unordered (only the longest
    input is used for history; call count is order-independent).
    """
    if not calls:
        return {"n_calls": 0, "error_outputs": 0, "retry_pairs": 0,
                "call_outlier": 0.0, "has_done_message": False,
                "abandoned_early": True, "expected_calls": 1.0}

    items = _history_items(calls)
    outputs = [i for i in items if i.get("type") in
               ("function_call_output", "custom_tool_call_output")]
    error_outputs = sum(bool(ERROR_RE.search(_text(i.get("output", "")))) for i in outputs)

    # A repeated function with nearly the same arguments within four tool calls
    # is a retry/backtrack.  Only consecutive-nearby repeats count, avoiding a
    # false alarm when a task legitimately revisits a tool much later.
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

    # If no empirical cohort median is supplied, longer requests get a modestly
    # larger expected budget.  It is a fallback, not a learned quality label.
    if expected_calls is None:
        prompt_chars = len(first_user_text(calls[0]))
        expected_calls = 3.0 + min(4.0, prompt_chars / 1500.0)
    expected_calls = max(float(expected_calls), 1.0)
    call_outlier = max(0.0, len(calls) / expected_calls - 1.0)
    # With several observed tool turns but no recoverable completion message,
    # call the trace potentially abandoned. The truly final response is absent,
    # so this is intentionally only a small penalty.
    abandoned = len(calls) >= 3 and bool(outputs) and not has_done
    return {"n_calls": len(calls), "error_outputs": error_outputs,
            "retry_pairs": retry_pairs, "call_outlier": call_outlier,
            "has_done_message": has_done, "abandoned_early": abandoned,
            "expected_calls": expected_calls}


def estimate_outcome(calls, expected_calls=None):
    """Estimate observed execution health on [0, 1]; higher is healthier.

    Penalties are capped, additive, and deliberately easy to challenge:
    errors <= .35, retries <= .25, excess calls <= .20, and an unresolved end
    <= .20.  A trajectory can recover from one bad tool call rather than being
    declared a total failure.
    """
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
        counts[cohort_key(calls)].append(len(calls)); all_counts.append(len(calls))
    global_median = sorted(all_counts)[len(all_counts) // 2] if all_counts else 1
    return {key: (sorted(values)[len(values) // 2] if len(values) >= 3 else global_median)
            for key, values in counts.items()}


# WHAT THIS PROXY DOES NOT CAPTURE (and how it can be wrong):
# - It cannot see the final answer, so a trace without a visible "done" may have
#   succeeded on its missing last call; conversely a confident done message may be wrong.
# - Tool errors/retries can be sensible exploration or successful recovery, and text
#   matching misses non-English or unrecognized errors.
# - More calls may reflect legitimate task complexity rather than poor model behavior;
#   coarse prompt/tool cohorts do not fully control task difficulty.
# - It measures execution traces, not user satisfaction, factuality, code correctness,
#   tool side effects, latency, or the quality/cost of unobserved model output.


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
