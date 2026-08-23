#!/usr/bin/env python3
"""Evaluate all challenge-required routing baselines on one cache-aware scale.

Policies are whole-trajectory commitments, so every generated route has zero internal
model switches. "Cheapest" and "strongest" preserve the logged provider family.
The random baseline is the mean of deterministic random policies matched as closely as
possible to the Smart Router's total estimated input cost.
"""
import argparse, csv, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline_router import route_trajectory as viktor_route
from cost_model import load_pricing, logged_route, trajectory_cost
from load_trajectories import group_trajectories, iter_requests
from outcome_evaluator import configure_estimator, estimate_route_quality, wilson_interval

CHEAP = {"claude": "claude-sonnet-5", "gpt": "gpt-5.6-luna"}
STRONG = {"claude": "claude-fable-5", "gpt": "gpt-5.6-sol"}

def family_model(logged, mapping):
    return mapping["claude"] if logged.startswith("claude") else mapping["gpt"]

def committed(calls, model): return [model] * len(calls)

def record(tid, calls, route, pricing):
    logged = logged_route(calls); cl, _ = trajectory_cost(calls, logged, pricing); cr, _ = trajectory_cost(calls, route, pricing)
    return {"trajectory": tid, "n_calls": len(calls), "logged_model": logged[0], "route": route,
            "cost_logged_usd": round(cl, 6), "cost_routed_usd": round(cr, 6),
            "switches": sum(route[i] != route[i-1] for i in range(1, len(route)))}

def summarize(name, records):
    probs = [estimate_route_quality(r["trajectory"], r["route"]) for r in records]
    q = sum(probs) / len(probs); lo, hi = wilson_interval(sum(probs), len(probs))
    return {"policy": name, "cost_usd": round(sum(r["cost_routed_usd"] for r in records), 6),
            "expected_success": round(q, 6), "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
            "rerouted_trajectories": sum(r["route"][0] != r["logged_model"] for r in records),
            "mid_trajectory_switches": sum(r["switches"] for r in records)}

def write_routes(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r) + "\n")

def random_at_matched_cost(base_records, cheap_records, target, trials, seed):
    """Return representative route plus Monte Carlo summaries near target cost."""
    rng = random.Random(seed); n = len(base_records); candidates = []
    for _ in range(trials):
        order = list(range(n)); rng.shuffle(order)
        chosen = set(); best = None
        for k in range(n + 1):
            records = [cheap_records[i] if i in chosen else base_records[i] for i in range(n)]
            summary = summarize("Random @ Smart cost", records)
            distance = abs(summary["cost_usd"] - target)
            if best is None or distance < best[0]: best = (distance, records, summary)
            if k < n: chosen.add(order[k])
        candidates.append(best)
    mean_cost = sum(x[2]["cost_usd"] for x in candidates) / trials
    mean_q = sum(x[2]["expected_success"] for x in candidates) / trials
    representative = min(candidates, key=lambda x: abs(x[2]["cost_usd"]-mean_cost) + abs(x[2]["expected_success"]-mean_q))[1]
    summary = summarize("Random @ Smart cost", representative)
    summary["cost_usd"] = round(mean_cost, 6); summary["expected_success"] = round(mean_q, 6)
    lo, hi = wilson_interval(mean_q * n, n); summary["ci95_low"], summary["ci95_high"] = round(lo, 6), round(hi, 6)
    summary["monte_carlo_trials"] = trials
    return representative, summary

def svg(rows, output):
    width, height, left, right, top, bottom = 1250, 720, 120, 70, 85, 125
    xmin, xmax = min(r["cost_usd"] for r in rows), max(r["cost_usd"] for r in rows)
    ymin, ymax = min(r["ci95_low"] for r in rows), max(r["ci95_high"] for r in rows)
    padx, pady = (xmax-xmin)*.08, (ymax-ymin)*.06; xmin-=padx; xmax+=padx; ymin=max(0,ymin-pady); ymax=min(1.02,ymax+pady)
    x=lambda v:left+(v-xmin)/(xmax-xmin)*(width-left-right); y=lambda v:height-bottom-(v-ymin)/(ymax-ymin)*(height-top-bottom)
    colors={"Observed / logged":"#150079","Always cheapest":"#FFBD9E","Always strongest":"#150079","Random @ Smart cost":"#77718f","Viktor length heuristic":"#999999","Smart opening gate":"#6748FD"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><rect width="100%" height="100%" fill="#fbfaff"/><style>text{{font-family:Inter,Segoe UI,sans-serif;fill:#150079}}.tick{{font-size:15px}}.name{{font-size:16px;font-weight:700}}</style><text x="625" y="42" text-anchor="middle" font-size="28" font-weight="700">Required policy comparison · one cache-aware scale</text>']
    for i in range(6):
        v=xmin+(xmax-xmin)*i/5; px=x(v); parts.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{height-bottom}" stroke="#150079" opacity=".09"/><text class="tick" x="{px}" y="{height-bottom+28}" text-anchor="middle">${v:.2f}</text>')
        q=ymin+(ymax-ymin)*i/5; py=y(q); parts.append(f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" stroke="#150079" opacity=".09"/><text class="tick" x="{left-15}" y="{py+5}" text-anchor="end">{q:.2f}</text>')
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#150079"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#150079"/>',f'<text x="625" y="{height-55}" text-anchor="middle" font-size="18">Estimated input cost (USD, cache-aware)</text>',f'<text transform="translate(32 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle" font-size="18">Expected task success</text>']
    offsets={"Observed / logged":(-15,-18,"end"),"Always cheapest":(15,36,"start"),"Always strongest":(-15,30,"end"),"Random @ Smart cost":(15,-24,"start"),"Viktor length heuristic":(-15,38,"end"),"Smart opening gate":(15,-28,"start")}
    for r in rows:
        px,py=x(r["cost_usd"]),y(r["expected_success"]); lo,hi=y(r["ci95_low"]),y(r["ci95_high"]); color=colors[r["policy"]]
        parts.append(f'<line x1="{px}" y1="{lo}" x2="{px}" y2="{hi}" stroke="{color}" stroke-width="3"/><line x1="{px-7}" y1="{lo}" x2="{px+7}" y2="{lo}" stroke="{color}" stroke-width="3"/><line x1="{px-7}" y1="{hi}" x2="{px+7}" y2="{hi}" stroke="{color}" stroke-width="3"/><circle cx="{px}" cy="{py}" r="10" fill="{color}" stroke="#150079"/>')
        dx,dy,anchor=offsets[r["policy"]]; parts.append(f'<text class="name" x="{px+dx}" y="{py+dy}" text-anchor="{anchor}">{r["policy"]}</text>')
    parts.append('<text x="625" y="685" text-anchor="middle" font-size="15">Whiskers: model-based 95% Wilson intervals · random policy matched to Smart cost over deterministic Monte Carlo trials</text></svg>')
    Path(output).write_text("".join(parts), encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("export",nargs="?",default="export"); p.add_argument("--smart",default="results/routes_smart.jsonl"); p.add_argument("--features",default="results/call_features.csv"); p.add_argument("--outcomes",default="results/outcomes.json"); p.add_argument("--trials",type=int,default=1000); p.add_argument("--seed",type=int,default=20260823); a=p.parse_args()
    configure_estimator(a.outcomes,a.features); groups=group_trajectories(r for _,_,r in iter_requests(a.export)); pricing=load_pricing()
    observed=[]; cheapest=[]; strongest=[]; viktor=[]
    for tid,calls in groups.items():
        logged=calls[0]["model"]
        observed.append(record(tid,calls,logged_route(calls),pricing))
        cheapest.append(record(tid,calls,committed(calls,family_model(logged,CHEAP)),pricing))
        strongest.append(record(tid,calls,committed(calls,family_model(logged,STRONG)),pricing))
        viktor.append(record(tid,calls,viktor_route(calls),pricing))
    smart=[json.loads(x) for x in open(a.smart,encoding="utf-8") if x.strip()]; smart.sort(key=lambda r:r["trajectory"])
    for records in (observed,cheapest,strongest,viktor): records.sort(key=lambda r:r["trajectory"])
    target=sum(r["cost_routed_usd"] for r in smart); random_records,random_summary=random_at_matched_cost(observed,cheapest,target,a.trials,a.seed)
    policies=[("Observed / logged",observed),("Always cheapest",cheapest),("Always strongest",strongest),("Random @ Smart cost",random_records),("Viktor length heuristic",viktor),("Smart opening gate",smart)]
    summaries=[]
    for name,records in policies:
        summary=random_summary if name=="Random @ Smart cost" else summarize(name,records); summary["policy"]=name; summaries.append(summary)
        write_routes(f"results/routes_{name.lower().replace(' / ','_').replace(' @ ','_').replace(' ','_')}.jsonl",records)
    fields=sorted({k for r in summaries for k in r},key=lambda k:(k!="policy",k))
    with open("results/required_baselines.csv","w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summaries)
    svg(summaries,"results/required_baselines.svg")
    print("\nRequired policy comparison")
    for r in summaries: print(f"  {r['policy']:24s} cost=${r['cost_usd']:.4f} quality={r['expected_success']:.3f} CI[{r['ci95_low']:.3f},{r['ci95_high']:.3f}] rerouted={r['rerouted_trajectories']}")
    print("NOTE: family-preserving cheapest/strongest; random is matched to Smart cost; all token costs estimated input-side.")
    print("wrote results/required_baselines.csv and results/required_baselines.svg")

if __name__=="__main__": main()
