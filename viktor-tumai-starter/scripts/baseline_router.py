#!/usr/bin/env python3
"""Feature-based heuristic router with cache-aware model-switch decisions.

The router uses only information available at each call: request size, tool-set
size, call depth, and tool outputs already present in the cumulative history.
It keeps the public ``route_trajectory(calls)`` interface used by the evaluator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from cost_model import (
    load_pricing,
    logged_route,
    price_of,
    shared_prefix_tokens,
    trajectory_cost,
)
from load_trajectories import est_tokens, group_trajectories, iter_requests
from outcomes import evaluate_trajectory_outcome


STRONG_MODELS = {"claude": "claude-opus-5", "gpt": "gpt-5.6-terra"}
CHEAP_MODEL = "gpt-5.6-luna"

# Inspectable thresholds intended for later tuning on a held-out set.
EXCEPTIONAL_TOOL_COUNT = 12
EXCEPTIONAL_TOOL_TOKENS = 4_000
SMALL_GROWTH_TOKENS = 750
SMALL_GROWTH_RATIO = 0.10


def _family(model: str) -> str:
    return "claude" if model.startswith("claude") else "gpt"


def strong_for(model: str) -> str:
    """Return the designated capable model in the logged model's family."""
    return STRONG_MODELS[_family(model)]


def cheap_for(model: str) -> str:
    """Return the globally cheapest model (argument retained for compatibility)."""
    del model
    return CHEAP_MODEL


def _history_has_new_error(
    calls: Sequence[Mapping[str, Any]], call_index: int
) -> bool:
    """Whether the output revealed since the preceding request has an error."""
    if call_index <= 0:
        return False
    current = evaluate_trajectory_outcome(
        pd.DataFrame.from_records(calls[: call_index + 1])
    )
    previous = evaluate_trajectory_outcome(
        pd.DataFrame.from_records(calls[:call_index])
    )
    return int(current["error_output_count"]) > int(previous["error_output_count"])


def build_call_features(calls: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Build chronological routing features from reconstructed requests."""
    records: list[dict[str, Any]] = []
    previous_tokens = 0

    for call_index, call in enumerate(calls):
        prompt_tokens = est_tokens(call.get("input", []))
        token_growth = prompt_tokens - previous_tokens if call_index else prompt_tokens
        tools = call.get("tools", [])
        tools_list = tools if isinstance(tools, list) else []
        tools_count = len(tools_list)
        tools_tokens = est_tokens(tools_list)
        records.append(
            {
                "call_index": call_index,
                "is_first_call": call_index == 0,
                "logged_model": str(call.get("model", "gpt-5.6-terra")),
                "prompt_tokens": prompt_tokens,
                "prompt_token_growth": max(0, token_growth),
                "prompt_growth_ratio": (
                    max(0, token_growth) / previous_tokens
                    if previous_tokens > 0
                    else 1.0
                ),
                "tools_count": tools_count,
                "tools_tokens": tools_tokens,
                "is_tool_heavy": (
                    tools_count >= EXCEPTIONAL_TOOL_COUNT
                    or tools_tokens >= EXCEPTIONAL_TOOL_TOKENS
                ),
                "previous_call_error": _history_has_new_error(calls, call_index),
            }
        )
        previous_tokens = prompt_tokens

    return pd.DataFrame.from_records(records)


def _small_prompt_growth(feature: pd.Series) -> bool:
    return bool(
        int(feature["prompt_token_growth"]) <= SMALL_GROWTH_TOKENS
        or float(feature["prompt_growth_ratio"]) <= SMALL_GROWTH_RATIO
    )


def _switch_is_economical(
    calls: Sequence[Mapping[str, Any]],
    call_index: int,
    current_model: str,
    candidate_model: str,
    pricing: Mapping[str, Sequence[float]],
) -> bool:
    """Check appended-token savings against the cache-reset penalty.

    Staying bills the shared prefix at the current model's cache-read rate;
    switching bills it at the candidate's uncached rate. A cheaper switch is
    allowed only when savings on appended tokens strictly exceed that penalty.
    """
    if candidate_model == current_model or call_index == 0:
        return True

    current_uncached, current_cached, _ = price_of(current_model, pricing)
    candidate_uncached, _, _ = price_of(candidate_model, pricing)
    if candidate_uncached >= current_uncached:
        # Capability upgrades are quality decisions rather than savings moves.
        return True

    call = calls[call_index]
    input_tokens = est_tokens(call.get("input", []))
    shared_tokens = min(
        shared_prefix_tokens(calls[call_index - 1], call), input_tokens
    )
    appended_tokens = max(0, input_tokens - shared_tokens)
    cache_reset_penalty = shared_tokens * max(
        0.0, candidate_uncached - current_cached
    )
    appended_token_savings = appended_tokens * max(
        0.0, current_uncached - candidate_uncached
    )
    return appended_token_savings > cache_reset_penalty


def route_trajectory(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    """Assign a model to each call with an inspectable heuristic decision tree."""
    if not calls:
        return []

    features = build_call_features(calls)
    pricing = load_pricing()
    route: list[str] = []

    for row_index, feature in features.iterrows():
        logged_model = str(feature["logged_model"])
        if bool(feature["is_first_call"]):
            candidate = strong_for(logged_model)
        elif bool(feature["is_tool_heavy"]) or bool(feature["previous_call_error"]):
            candidate = strong_for(logged_model)
        elif int(feature["call_index"]) > 2 and _small_prompt_growth(feature):
            candidate = CHEAP_MODEL
        else:
            candidate = route[-1]

        if route and candidate != route[-1] and not _switch_is_economical(
            calls,
            int(row_index),
            route[-1],
            candidate,
            pricing,
        ):
            candidate = route[-1]
        route.append(candidate)

    return route


def main() -> None:
    export = sys.argv[1] if len(sys.argv) > 1 else "export"
    pricing = load_pricing()
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    Path("results").mkdir(exist_ok=True)

    total_logged = 0.0
    total_routed = 0.0
    with Path("results/routes.jsonl").open("w", encoding="utf-8") as output:
        for key, calls in groups.items():
            logged = logged_route(calls)
            routed = route_trajectory(calls)
            logged_cost, _ = trajectory_cost(calls, logged, pricing)
            routed_cost, _ = trajectory_cost(calls, routed, pricing)
            total_logged += logged_cost
            total_routed += routed_cost
            output.write(
                json.dumps(
                    {
                        "trajectory": key,
                        "n_calls": len(calls),
                        "logged_model": logged[0],
                        "route": routed,
                        "cost_logged_usd": round(logged_cost, 6),
                        "cost_routed_usd": round(routed_cost, 6),
                        "switches": sum(
                            routed[i] != routed[i - 1]
                            for i in range(1, len(routed))
                        ),
                    }
                )
                + "\n"
            )

    change = (total_routed / total_logged - 1.0) if total_logged else 0.0
    print(f"logged cost (estimated input tokens): ${total_logged:,.4f}")
    print(f"routed cost: ${total_routed:,.4f} ({change:+.1%}, cache-aware)")
    print("NOTE: output cost is excluded because final outputs and usage are absent.")
    print("wrote results/routes.jsonl")


if __name__ == "__main__":
    main()
