#!/usr/bin/env python3
"""Plot conservative, matched off-policy cost--quality frontiers.

The old quality placeholder ("fraction kept on logged model") has been
replaced with the structural execution-health score from outcome_signal.py.
When a policy changes a route, the counterfactual score comes from the nearest
comparable logged trajectory that actually ran on the proposed model. A route
without a close match is not adopted in the headline estimate; this makes lack
of common support explicit instead of silently extrapolating.

Usage:
  python scripts/plot_frontier.py results/baseline_routes.jsonl \
      results/learned_routes.jsonl export/

The two route-file arguments may be omitted only for backward compatibility;
then results/routes.jsonl is plotted as a single policy. Writes
results/frontier.csv and results/frontier.png (if matplotlib is installed).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

from cost_model import load_pricing, price_of, trajectory_cost
from load_trajectories import group_trajectories, iter_requests
from offpolicy_eval import build_records, match_distance, robust_scales


CALIPER = 1.25  # same normalized-distance common-support rule as offpolicy_eval.py


def read_routes(path):
    return {r["trajectory"]: r for r in (json.loads(line) for line in open(path))}


def route_model(record):
    """All policies here must preserve the one-model-per-trajectory premise."""
    models = set(record["route"])
    if len(models) != 1:
        raise ValueError(f"{record['trajectory']} switches models; this frontier requires whole-trajectory routes")
    return record["route"][0]


def make_reference(groups):
    """Observed outcomes and features indexed by trajectory/model for matching."""
    records = build_records(groups)
    scales = robust_scales(records)
    by_key = {r["trajectory"]: r for r in records}
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    return by_key, by_model, scales


def quality(record, reference, by_model, scales):
    """Return (structural outcome estimate, matched?, distance).

    Kept routes use their actually observed outcome. Changed routes use one
    nearest-neighbour counterfactual: the real outcome signal of the closest
    trace logged on the proposed model. This is matching OPE, not a claim that
    the proposed route was executed.
    """
    observed = reference[record["trajectory"]]
    proposed = route_model(record)
    if proposed == observed["model"]:
        return observed["outcome"], True, 0.0
    candidates = by_model.get(proposed, [])
    if not candidates:
        return observed["outcome"], False, None
    distance, match = min(((match_distance(observed, c, scales), c) for c in candidates),
                          key=lambda pair: pair[0])
    if distance > CALIPER:
        return observed["outcome"], False, distance
    return match["outcome"], True, distance


def comparator_routes(groups, pricing):
    """Optional anchors: cheapest, most-expensive, deterministic random model."""
    models = sorted({calls[0]["model"] for calls in groups.values()
                     if len({c["model"] for c in calls}) == 1})
    cheapest = min(models, key=lambda m: price_of(m, pricing)[0])
    expensive = max(models, key=lambda m: price_of(m, pricing)[0])
    output = {"always-cheapest": {}, "always-most-expensive": {}, "random": {}}
    for key, calls in groups.items():
        if len({c["model"] for c in calls}) != 1:
            continue
        logged = calls[0]["model"]
        random_model = models[int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(models)]
        for label, model in (("always-cheapest", cheapest), ("always-most-expensive", expensive),
                             ("random", random_model)):
            logged_cost, _ = trajectory_cost(calls, [logged] * len(calls), pricing)
            routed_cost, _ = trajectory_cost(calls, [model] * len(calls), pricing)
            output[label][key] = {"trajectory": key, "n_calls": len(calls), "logged_model": logged,
                                  "route": [model] * len(calls), "cost_logged_usd": logged_cost,
                                  "cost_routed_usd": routed_cost, "switches": 0}
    return output


def frontier(policy, routes, reference, by_model, scales):
    """Adopt each route proposal in descending savings order, when supported."""
    recs = [r for key, r in routes.items() if key in reference]
    scored = []
    for r in recs:
        q, supported, distance = quality(r, reference, by_model, scales)
        r = dict(r)
        r["estimated_quality"] = q; r["supported"] = supported; r["match_distance"] = distance
        r["savings"] = r["cost_logged_usd"] - r["cost_routed_usd"]
        scored.append(r)
    # A partial adoption spends common support on the greatest savings first.
    # Unsupported proposals remain logged, rather than receiving guessed quality.
    scored.sort(key=lambda r: r["savings"], reverse=True)
    observed_total = sum(reference[r["trajectory"]]["outcome"] for r in scored)
    rows = []
    for tenth in range(11):
        requested = int(round(tenth * len(scored) / 10))
        selected = set(r["trajectory"] for r in scored[:requested])
        cost = 0.0; total_quality = 0.0; adopted = unsupported = 0; distances = []
        for r in scored:
            use_route = r["trajectory"] in selected and r["supported"]
            if r["trajectory"] in selected and not r["supported"]:
                unsupported += 1
            if use_route:
                adopted += 1; cost += r["cost_routed_usd"]; total_quality += r["estimated_quality"]
                if r["match_distance"] is not None: distances.append(r["match_distance"])
            else:
                cost += r["cost_logged_usd"]; total_quality += reference[r["trajectory"]]["outcome"]
        rows.append({"policy": policy, "adopt_frac_requested": tenth / 10,
                     "adopted_supported": adopted, "unsupported_proposals": unsupported,
                     "cost_usd": round(cost, 6),
                     "estimated_execution_health": round(total_quality / len(scored), 4),
                     "mean_match_distance": round(sum(distances) / len(distances), 4) if distances else "",
                     "baseline_observed_execution_health": round(observed_total / len(scored), 4)})
    return rows


def main():
    if len(sys.argv) >= 4:
        baseline_path, learned_path, export = map(Path, sys.argv[1:4])
        policies = {"baseline": read_routes(baseline_path), "learned": read_routes(learned_path)}
    else:
        export = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("export")
        source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/routes.jsonl")
        policies = {"policy": read_routes(source)}
    groups = group_trajectories(r for _, _, r in iter_requests(export))
    reference, by_model, scales = make_reference(groups)
    policies.update(comparator_routes(groups, load_pricing()))
    rows = []
    for name, routes in policies.items():
        rows.extend(frontier(name, routes, reference, by_model, scales))
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    out = out_dir / "frontier.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {out}")
    for name in policies:
        endpoint = [r for r in rows if r["policy"] == name][-1]
        print(f"{name}: cost=${endpoint['cost_usd']:.4f}, health={endpoint['estimated_execution_health']:.3f}, "
              f"supported={endpoint['adopted_supported']}, unsupported={endpoint['unsupported_proposals']}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2, 4.8))
        colors = {"baseline": "#7F8C8D", "learned": "#6748FD", "always-cheapest": "#E67E22",
                  "always-most-expensive": "#C0392B", "random": "#16A085", "policy": "#6748FD"}
        for name in policies:
            series = [r for r in rows if r["policy"] == name]
            plt.plot([r["cost_usd"] for r in series], [r["estimated_execution_health"] for r in series],
                     "o-", label=name, color=colors.get(name))
        plt.xlabel("input-side cost (USD; estimated tokens, cache-aware)")
        plt.ylabel("estimated execution health (matched OPE proxy)")
        plt.title("Cost–quality frontier: conservative matched off-policy estimate")
        plt.legend(); plt.tight_layout()
        png = out_dir / "frontier.png"; plt.savefig(png, dpi=160)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed — wrote CSV only")
    print("NOTE: changed routes are evaluated only with a close observed model match; output correctness/cost remains unobserved.")


if __name__ == "__main__":
    main()
