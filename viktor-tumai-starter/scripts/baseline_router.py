#!/usr/bin/env python3
"""Feature-based heuristic router with cache-aware model-switch decisions.

The router uses only information available at each call: request size, tool-set
size, call depth, and tool outputs already present in the cumulative history.
It keeps the public ``route_trajectory(calls)`` interface used by the evaluator.
"""

from __future__ import annotations

import json
import math
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


MODEL_TIERS = {
    "claude": {
        "strong": "claude-opus-5",
        "mid": "claude-sonnet-5",
        # There is no low-cost Claude tier in the supplied price schedule.
        "cheap": "gpt-5.6-luna",
    },
    "gpt": {
        "strong": "gpt-5.6-sol",
        "mid": "gpt-5.6-terra",
        "cheap": "gpt-5.6-luna",
    },
}

PROMPT_SIZE_ANCHOR = 20_000
TOKEN_GROWTH_ANCHOR = 2_000
TOOL_RATIO_ANCHOR = 0.25
DEFAULT_AGGRESSIVENESS = 0.5


def _family(model: str) -> str:
    return "claude" if model.startswith("claude") else "gpt"


def strong_for(model: str) -> str:
    """Return the designated capable model in the logged model's family."""
    return MODEL_TIERS[_family(model)]["strong"]


def mid_for(model: str) -> str:
    """Return the balanced model in the logged model's family."""
    return MODEL_TIERS[_family(model)]["mid"]


def cheap_for(model: str) -> str:
    """Return the economical tier associated with the logged family."""
    return MODEL_TIERS[_family(model)]["cheap"]


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
        total_context_tokens = prompt_tokens + tools_tokens
        tool_token_ratio = (
            tools_tokens / total_context_tokens if total_context_tokens else 0.0
        )
        size_score = min(
            1.0, math.log1p(prompt_tokens) / math.log1p(PROMPT_SIZE_ANCHOR)
        )
        growth_score = min(1.0, max(0, token_growth) / TOKEN_GROWTH_ANCHOR)
        tool_score = min(1.0, tool_token_ratio / TOOL_RATIO_ANCHOR)
        previous_call_error = _history_has_new_error(calls, call_index)
        # Growth and tool density dominate because they are the strongest
        # online signals that a continuation requires fresh reasoning.
        complexity_score = min(
            1.0,
            0.15 * size_score + 0.50 * growth_score + 0.35 * tool_score,
        )
        if previous_call_error:
            complexity_score = 1.0
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
                "tool_token_ratio": tool_token_ratio,
                "size_score": size_score,
                "growth_score": growth_score,
                "tool_score": tool_score,
                "complexity_score": complexity_score,
                "previous_call_error": previous_call_error,
            }
        )
        previous_tokens = prompt_tokens

    return pd.DataFrame.from_records(records)


def _tier_for_score(score: float, aggressiveness: float) -> str:
    """Map continuous utility to a tier with smoothly moving boundaries."""
    cheap_threshold = 0.12 + 0.58 * aggressiveness
    strong_threshold = 0.52 + 0.43 * aggressiveness
    if score < cheap_threshold:
        return "cheap"
    if score < strong_threshold:
        return "mid"
    return "strong"


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


def _route_at_level(
    calls: Sequence[Mapping[str, Any]],
    features: pd.DataFrame,
    aggressiveness: float,
    pricing: Mapping[str, Sequence[float]],
) -> list[str]:
    """Build a route at one exact point on the utility sweep."""
    route: list[str] = []

    for row_index, feature in features.iterrows():
        logged_model = str(feature["logged_model"])
        tier = _tier_for_score(float(feature["complexity_score"]), aggressiveness)
        candidate = MODEL_TIERS[_family(logged_model)][tier]

        # Every downward switch must pay for its inferred cache reset using
        # strictly larger marginal input savings. Upgrades remain quality-led.
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


def _sweep_levels(features: pd.DataFrame, aggressiveness: float) -> list[float]:
    """Return all decision breakpoints up to the requested sweep position."""
    levels = {0.0, aggressiveness}
    for score in features["complexity_score"].astype(float):
        for breakpoint in ((score - 0.12) / 0.58, (score - 0.52) / 0.43):
            if 0.0 < breakpoint <= aggressiveness:
                levels.add(min(aggressiveness, breakpoint + 1e-9))
    return sorted(levels)


def route_trajectory(
    calls: Sequence[Mapping[str, Any]],
    aggressiveness: float = DEFAULT_AGGRESSIVENESS,
) -> list[str]:
    """Route calls using continuous utility and a monotone cost envelope.

    Higher aggressiveness admits progressively more downgrade candidates. At
    every decision breakpoint, the complete cache-aware trajectory cost is
    evaluated. Returning the cheapest route admitted so far guarantees that
    cost can never rise as aggressiveness increases.
    """
    if not calls:
        return []
    if not 0.0 <= aggressiveness <= 1.0:
        raise ValueError("aggressiveness must be between 0.0 and 1.0")

    features = build_call_features(calls)
    pricing = load_pricing()
    best_route: list[str] | None = None
    best_cost = math.inf

    for level in _sweep_levels(features, aggressiveness):
        candidate = _route_at_level(calls, features, level, pricing)
        candidate_cost, _ = trajectory_cost(calls, candidate, pricing)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_route = candidate

    assert best_route is not None
    return best_route


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
