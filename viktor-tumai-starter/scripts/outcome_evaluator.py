#!/usr/bin/env python3
"""Detect task completion and estimate matched off-policy route quality.

The estimator is observational, not proof of counterfactual success. It matches the
logged trajectory-level treatment on early features, uses Beta(1,1) smoothing, and
backs off to coarser strata when exact cells are sparse.
"""
from __future__ import annotations

import argparse, csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from load_trajectories import group_trajectories, iter_requests

POSITIVE = re.compile(
    r"\b(success(?:ful(?:ly)?)?|completed?|done|finished|verified|passed|resolved|"
    r"created|updated|moved|fixed|deployed|saved|sent|all tests? pass(?:ed)?)\b", re.I)
NEGATIVE = re.compile(
    r"\b(fail(?:ed|ure)?|error|exception|unable|cannot|can't|incomplete|not completed|"
    r"not verified|timed? out|timeout|blocked|aborted|permission denied)\b", re.I)
CHEAP_MODELS = {"gpt-5.6-luna", "claude-sonnet-5"}
PREMIUM_HINTS = ("fable", "opus", "gpt-5.6-sol")
MIN_SUPPORT = 5
_STATE: dict[str, Any] | None = None


def item_type(item):
    return item.get("type", "message")


def message_text(item):
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(str(p.get("text", "")) for p in content
                    if isinstance(p, dict) and p.get("type") in {"input_text", "output_text", "text"})


def terminal_evidence(calls):
    """Last observable assistant message or function result in the final request."""
    for item in reversed(calls[-1]["input"]):
        if item.get("role") == "assistant":
            return "assistant", message_text(item)
        if item_type(item) in {"function_call_output", "custom_tool_call_output"}:
            value = item.get("output", "")
            return "tool_output", value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return "none", ""


def detect_success(calls):
    """Conservative binary heuristic; negative terminal evidence takes precedence."""
    _, text = terminal_evidence(calls)
    if not text.strip() or NEGATIVE.search(text):
        return False
    if POSITIVE.search(text):
        return True
    lowered = text.lower()
    return bool(re.search(r'"(?:success|ok)"\s*:\s*true', lowered)
                or re.search(r'"(?:exit_?code|return_?code)"\s*:\s*0\b', lowered))


def build_outcomes(export_dir):
    groups = group_trajectories(req for _, _, req in iter_requests(export_dir))
    return {tid: {"logged_model": calls[0]["model"], "success": detect_success(calls)}
            for tid, calls in groups.items()}


def model_tier(model):
    if model in CHEAP_MODELS:
        return "cheap"
    if any(hint in model for hint in PREMIUM_HINTS):
        return "premium"
    return "standard"


def model_family(model):
    return "claude" if model.startswith("claude") else ("gpt" if model.startswith("gpt") else "other")


def length_bucket(n):
    return "short" if n <= 3 else ("medium" if n <= 6 else "long")


def load_early_features(path):
    features = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["call_index"]) == 1:
                n = int(row["trajectory_calls"])
                features[row["trajectory"]] = {
                    "task_category": row["task_category"],
                    "has_input_image": int(row["has_input_image"]),
                    "has_reasoning": int(row["has_reasoning"]),
                    "trajectory_calls": n,
                    "length_bucket": length_bucket(n),
                }
    return features


def _make_state(outcomes_path, features_path):
    outcomes = json.loads(Path(outcomes_path).read_text(encoding="utf-8"))
    features = load_early_features(features_path)
    observations = []
    for tid, outcome in outcomes.items():
        if tid in features:
            observations.append({"trajectory": tid, "success": int(outcome["success"]),
                                 "model": outcome["logged_model"],
                                 "tier": model_tier(outcome["logged_model"]), **features[tid]})
    return {"outcomes": outcomes, "features": features, "observations": observations}


def configure_estimator(outcomes_path=ROOT / "results" / "outcomes.json",
                        features_path=ROOT / "results" / "call_features.csv"):
    global _STATE
    _STATE = _make_state(outcomes_path, features_path)


def _state():
    global _STATE
    if _STATE is None:
        configure_estimator()
    return _STATE


def _matched_rate(trajectory_id, tier):
    state, target = _state(), _state()["features"].get(trajectory_id)
    levels = [
        ("exact", ("task_category", "has_input_image", "has_reasoning", "length_bucket")),
        ("drop_length", ("task_category", "has_input_image", "has_reasoning")),
        ("category_image", ("task_category", "has_input_image")),
        ("category", ("task_category",)),
        ("tier_global", ()),
    ]
    if target is None:
        levels = [("tier_global", ())]
    fallback = None
    for level, fields in levels:
        values = [o["success"] for o in state["observations"]
                  if o["tier"] == tier and (target is None or all(o[f] == target[f] for f in fields))]
        candidate = ((sum(values) + 1) / (len(values) + 2), len(values), level) if values else None
        fallback = fallback or candidate
        if len(values) >= MIN_SUPPORT or level == "tier_global":
            return candidate or fallback or (0.5, 0, "no_support")
    return fallback or (0.5, 0, "no_support")


def estimate_route_quality(trajectory_id, proposed_route):
    """Expected trajectory success in [0,1]; mixed routes are an exploratory blend."""
    state = _state()
    if not proposed_route:
        return 0.5
    outcome = state["outcomes"].get(trajectory_id)
    if outcome and all(m == outcome["logged_model"] for m in proposed_route):
        return float(outcome["success"])
    rates = [_matched_rate(trajectory_id, model_tier(model))[0] for model in proposed_route]
    return max(0.0, min(1.0, math.exp(sum(math.log(max(r, 1e-9)) for r in rates) / len(rates))))


def wilson(successes, total, z=1.96):
    if not total:
        return 0.0, 1.0
    p, den = successes / total, 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return center-half, center+half


def print_summary(state):
    obs, cells = state["observations"], defaultdict(Counter)
    successes = sum(o["success"] for o in obs)
    print("\n=== Task completion and matched-treatment summary ===")
    print(f"trajectories={len(obs)}  detected_success={successes} ({successes/max(1,len(obs)):.1%})")
    print("\nObserved success by model tier (95% Wilson interval; not causal):")
    for tier in ("cheap", "standard", "premium"):
        rows = [o for o in obs if o["tier"] == tier]
        s, n = sum(o["success"] for o in rows), len(rows)
        lo, hi = wilson(s, n)
        print(f"  {tier:8s} {s}/{n} = {s/max(1,n):.1%}  CI [{lo:.1%}, {hi:.1%}]")
    print("\nObserved success by provider family and logged model:")
    for label, selector in (
        *[(family, lambda o, family=family: model_family(o["model"]) == family)
          for family in sorted({model_family(o["model"]) for o in obs})],
        *[(model, lambda o, model=model: o["model"] == model)
          for model in sorted({o["model"] for o in obs})],
    ):
        rows = [o for o in obs if selector(o)]
        s, n = sum(o["success"] for o in rows), len(rows)
        print(f"  {label:20s} {s}/{n} = {s/max(1,n):.1%}")
    for row in obs:
        cells[(row["task_category"], row["has_input_image"],
               row["has_reasoning"], row["length_bucket"])][row["tier"]] += 1
    overlap = [(k, c) for k, c in cells.items() if c["cheap"] >= MIN_SUPPORT and c["premium"] >= MIN_SUPPORT]
    print(f"\nExact matched cells with >= {MIN_SUPPORT} cheap AND premium trajectories: {len(overlap)}/{len(cells)}")
    for key, counts in sorted(cells.items()):
        usable = counts["cheap"] >= MIN_SUPPORT and counts["premium"] >= MIN_SUPPORT
        print(f"  {key}: cheap={counts['cheap']} premium={counts['premium']} "
              f"standard={counts['standard']}  {'usable' if usable else 'SPARSE'}")
    if len(overlap) < len(cells):
        print("WARNING: sparse cells lack common support; estimates back off to coarser")
        print("strata and then tier-wide rates. Wide intervals preclude a confident 95% claim.")
    print("CAUSAL WARNING: recorded-feature matching cannot remove assignment confounding,")
    print("completion-label error, or unobserved mixed-route interactions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="export")
    parser.add_argument("features", nargs="?", default="results/call_features.csv")
    parser.add_argument("--output", default="results/outcomes.json")
    args = parser.parse_args()
    outcomes, output = build_outcomes(args.export), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(outcomes, indent=2, sort_keys=True), encoding="utf-8")
    configure_estimator(output, args.features)
    print_summary(_state())
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
