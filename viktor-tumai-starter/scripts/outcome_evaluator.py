#!/usr/bin/env python3
"""Completion labels plus CEM-based off-policy quality evaluation."""
import argparse, csv, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from load_trajectories import group_trajectories, iter_requests
from cost_model import load_pricing, price_of, shared_prefix_tokens, trajectory_cost

CHEAP = {"gpt-5.6-luna", "claude-sonnet-5"}
PREMIUM = ("fable", "opus", "gpt-5.6-sol")
POS = re.compile(r"\b(success|successful|completed?|done|finished|verified|passed|resolved|created|updated|moved|fixed|saved|sent)\b", re.I)
NEG = re.compile(r"\b(fail|failed|failure|error|exception|unable|cannot|incomplete|timeout|blocked|aborted|permission denied)\b", re.I)
_STATE = None


def tier(model):
    if model in CHEAP: return "cheap"
    if any(x in model for x in PREMIUM): return "premium"
    return "standard"


def text(item):
    value = item.get("output", item.get("content", ""))
    if isinstance(value, str): return value
    if isinstance(value, list): return " ".join(str(p.get("text", "")) for p in value if isinstance(p, dict))
    return json.dumps(value, ensure_ascii=False)


def terminal(calls):
    for item in reversed(calls[-1]["input"]):
        if item.get("role") == "assistant" or item.get("type") in {"function_call_output", "custom_tool_call_output"}:
            return text(item)
    return ""


def success(calls):
    value = terminal(calls)
    if not value or NEG.search(value): return False
    return bool(POS.search(value) or re.search(r'"(?:ok|success)"\s*:\s*true|"(?:exit_?code|return_?code)"\s*:\s*0', value, re.I))


def build_outcomes(export):
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    return {k: {"logged_model": v[0]["model"], "success": success(v)} for k, v in groups.items()}


def bucket(n): return "short" if n <= 3 else ("medium" if n <= 6 else "long")


def early_features(path):
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["call_index"]) == 1:
                result[row["trajectory"]] = (row["task_category"], int(row["has_input_image"]), bucket(int(row["trajectory_calls"])))
    return result


def configure_estimator(outcomes_path=ROOT/"results/outcomes.json", features_path=ROOT/"results/call_features.csv"):
    global _STATE
    outcomes = json.loads(Path(outcomes_path).read_text(encoding="utf-8")); features = early_features(features_path)
    obs = [{"id": k, "model": v["logged_model"], "tier": tier(v["logged_model"]),
            "success": int(v["success"]), "stratum": features[k]} for k, v in outcomes.items() if k in features]
    strata = defaultdict(list)
    for o in obs: strata[o["stratum"]].append(o)
    overlap = {s for s, rows in strata.items() if {"cheap", "premium"} <= {r["tier"] for r in rows}}
    # ATE CEM weights: each treatment tier receives half of each overlapping stratum.
    weighted = []
    for s in overlap:
        rows = [r for r in strata[s] if r["tier"] in {"cheap", "premium"}]
        counts = Counter(r["tier"] for r in rows); total = len(rows)
        for r in rows:
            weighted.append((r, total / (2 * counts[r["tier"]])))
    _STATE = {"outcomes": outcomes, "features": features, "obs": obs, "strata": strata,
              "overlap": overlap, "weighted": weighted}


def state():
    if _STATE is None: configure_estimator()
    return _STATE


def cem_diagnostics():
    s = state(); weights = [w for _, w in s["weighted"]]
    ess = (sum(weights)**2 / sum(w*w for w in weights)) if weights else 0.0
    usable = sum(len(s["strata"][x]) for x in s["overlap"])
    return {"strata_total": len(s["strata"]), "strata_overlap": len(s["overlap"]),
            "trajectories_total": len(s["obs"]), "trajectories_overlap": usable, "ess": ess}


def cem_rate(target_tier, stratum=None):
    s = state()
    exact = [r["success"] for r in s["strata"].get(stratum, []) if r["tier"] == target_tier] if stratum in s["overlap"] else []
    if exact: return (sum(exact)+1)/(len(exact)+2), len(exact), "exact_cem"
    rows = [(r, w) for r, w in s["weighted"] if r["tier"] == target_tier]
    if rows:
        return (sum(r["success"]*w for r,w in rows)+1)/(sum(w for _,w in rows)+2), len(rows), "weighted_cem"
    raw = [r["success"] for r in s["obs"] if r["tier"] == target_tier]
    return ((sum(raw)+1)/(len(raw)+2) if raw else .5), len(raw), "unmatched_fallback"


def estimate_route_quality(trajectory_id, proposed_route):
    s = state(); outcome = s["outcomes"].get(trajectory_id)
    if not proposed_route: return .5
    if outcome and all(m == outcome["logged_model"] for m in proposed_route): return float(outcome["success"])
    rates = [cem_rate(tier(m), s["features"].get(trajectory_id))[0] for m in proposed_route]
    return math.exp(sum(math.log(max(1e-9, r)) for r in rates)/len(rates))


def wilson_interval(expected_successes, n, z=1.96):
    """Wilson interval using fractional expected successes (model-based approximation)."""
    if not n: return 0.0, 1.0
    p = expected_successes/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)


def gamma_to_flip(candidate, comparator):
    """Odds-ratio sensitivity: Gamma needed to reduce candidate to comparator."""
    if candidate <= comparator: return 1.0
    odds_c = candidate/max(1e-9, 1-candidate); odds_b = comparator/max(1e-9, 1-comparator)
    return max(1.0, odds_c/max(1e-9, odds_b))


def cache_repricing_sanity(export):
    pricing = load_pricing(); groups = group_trajectories(r for _,_,r in iter_requests(export)); checked = 0
    for calls in groups.values():
        model = calls[0]["model"]; route = [model]*len(calls)
        actual, uncached = trajectory_cost(calls, route, pricing)
        expected_u = expected_usd = 0.0
        for i, call in enumerate(calls):
            from load_trajectories import est_tokens
            inp = est_tokens(call["input"]); cached = 0 if i == 0 else min(inp, shared_prefix_tokens(calls[i-1], call))
            u = inp-cached; pu, pc, _ = price_of(model, pricing)
            expected_u += u; expected_usd += (u*pu+cached*pc)/1e6
        assert uncached == expected_u and abs(actual-expected_usd) < 1e-12
        checked += 1
    return checked


def report():
    d=cem_diagnostics(); print(f"CEM strata overlap={d['strata_overlap']}/{d['strata_total']} trajectories={d['trajectories_overlap']}/{d['trajectories_total']} ESS={d['ess']:.2f}")
    for t in ("cheap","premium"): print(f"CEM {t} rate={cem_rate(t)[0]:.3f} support={cem_rate(t)[1]} source={cem_rate(t)[2]}")
    print("FAILURE BOUNDS: sparse overlap, observational assignment confounding, heuristic terminal-label noise.")


def main():
    p=argparse.ArgumentParser(); p.add_argument("export",nargs="?",default="export"); p.add_argument("features",nargs="?",default="results/call_features.csv"); p.add_argument("--output",default="results/outcomes.json"); a=p.parse_args()
    out=build_outcomes(a.export); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8")
    configure_estimator(a.output,a.features); report(); print(f"cache repricing sanity: PASS ({cache_repricing_sanity(a.export)} constant-model trajectories)"); print(f"wrote {a.output}")

if __name__ == "__main__": main()
