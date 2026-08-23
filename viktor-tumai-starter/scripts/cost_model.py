#!/usr/bin/env python3
"""Cache-aware cost model on ESTIMATED tokens.

The export has no `usage` field: every token count here is estimated as
serialized chars / 4 (see load_trajectories.est_tokens). The cache trap still
applies: providers cache the shared input prefix across consecutive calls of a
task, and a model switch resets that cache — the first call after a switch
pays the uncached rate for the whole prefix.

Cached share of call i is estimated as the token size of the item-level prefix
that call i shares with call i-1 of the same trajectory.

Model ids are real models with real prices. DEFAULT_PRICING below is the
official per-model schedule (per 1M tokens). Note fable-5 is the premium tier
(most expensive), and gpt-5.6-luna the cheapest. Cached input uses the
cache-READ rate; the cache-write premium and long-context/fast tiers are not
modeled (input-side estimates only). Unknown ids fall back to their family
rate by longest prefix; scripts/pricing.json overrides everything.
"""
import json
from pathlib import Path
from load_trajectories import est_tokens

DEFAULT_PRICING = {  # per 1M tokens: [uncached_input, cached_input(read), output] — official rates
    "claude-opus-5": [5.00, 0.50, 25.00],
    "claude-opus-4-8": [5.00, 0.50, 25.00],
    "claude-opus-4-6": [5.00, 0.50, 25.00],
    "claude-sonnet-5": [2.00, 0.20, 10.00],
    "claude-sonnet-4-6": [3.00, 0.30, 15.00],
    "claude-fable-5": [10.00, 1.00, 50.00],  # premium tier — most expensive model in the log
    "gpt-5.6-sol": [5.00, 0.50, 30.00],
    "gpt-5.6-terra": [2.00, 0.20, 12.00],
    "gpt-5.6-luna": [0.20, 0.02, 1.20],      # cheapest model in the log
    "claude-opus": [5.00, 0.50, 25.00],      # family fallbacks for unseen generations
    "claude-sonnet": [2.00, 0.20, 10.00],
    "claude-fable": [10.00, 1.00, 50.00],
    "gpt-5.6": [2.00, 0.20, 12.00],
    "_default": [2.00, 0.20, 10.00],
}

def load_pricing():
    p = Path(__file__).parent / "pricing.json"
    return json.loads(p.read_text()) if p.exists() else DEFAULT_PRICING

def price_of(model, pricing):
    if model in pricing: return pricing[model]
    for prefix in sorted(pricing, key=len, reverse=True):  # longest family prefix wins
        if prefix != "_default" and model.startswith(prefix): return pricing[prefix]
    return pricing["_default"]

def shared_prefix_tokens(prev_req, req):
    """Estimated tokens of the item-level input prefix shared with the previous call."""
    shared = 0
    for a, b in zip(prev_req["input"], req["input"]):
        if a == b: shared += est_tokens(a)
        else: break
    return shared

def trajectory_cost(calls, route, pricing=None):
    """Cost of a reconstructed trajectory (calls ordered by input length) if call i
    had been served by route[i]. Cache-aware: the shared prefix is billed at the
    cached rate only when route[i] == route[i-1]. Output tokens are unknowable
    (no outputs in the export) and are NOT included — say so when you quote numbers.
    Returns (usd, uncached_input_tokens_est)."""
    pricing = pricing or load_pricing()
    usd, uncached_total = 0.0, 0
    for i, c in enumerate(calls):
        inp = est_tokens(c["input"])
        cached = shared_prefix_tokens(calls[i - 1], c) if (i > 0 and route[i] == route[i - 1]) else 0
        cached = min(cached, inp)
        uncached = inp - cached
        pu, pc, _ = price_of(route[i], pricing)
        usd += (uncached * pu + cached * pc) / 1e6
        uncached_total += uncached
    return usd, uncached_total

def logged_route(calls): return [c["model"] for c in calls]
