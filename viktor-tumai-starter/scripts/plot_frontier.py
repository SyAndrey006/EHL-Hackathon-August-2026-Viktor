#!/usr/bin/env python3
import sys, json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    # Only process arguments that are explicitly .jsonl files
    route_files = [arg for arg in sys.argv[1:] if arg.endswith('.jsonl')]
    
    if not route_files:
        print("No .jsonl files provided.")
        return

    plt.figure(figsize=(9, 6), dpi=150)
    # Grey for Baseline, Purple for Learned, Green for Confidence
    colors = ['#808080', '#6748FD', '#2ca02c'] 
    markers = ['o', 's', '^']
    
    all_rows = []
    
    for idx, src in enumerate(route_files):
        name = Path(src).stem.replace('_routes', '').replace('routes', 'Baseline').title()
        recs = [json.loads(line) for line in open(src)]
        total_calls = sum(r["n_calls"] for r in recs)
        
        # Sort by cost for the adoption sweep (safest bets first)
        by_size = sorted(recs, key=lambda r: r.get("cost_logged_usd", 0))
        rows = []
        
        for frac in [i / 10 for i in range(11)]:
            adopt = set(id(r) for r in by_size[: int(round(frac * len(recs)))])
            
            cost = 0.0
            kept = 0.0
            
            for r in recs:
                if id(r) in adopt:
                    cost += r.get("cost_routed_usd", r.get("cost_logged_usd", 0))
                    # Extract the predicted quality from the routers, or default to baseline avg
                    q = r.get("predicted_proposal_outcome", r.get("predicted_routed_outcome", 0.97))
                else:
                    cost += r.get("cost_logged_usd", 0)
                    q = r.get("predicted_logged_outcome", 0.97)
                        
                kept += q * r["n_calls"]
            
            avg_q = kept / total_calls
            rows.append({"strategy": name, "adopt_frac": frac, "cost_usd": round(cost, 4), "quality": round(avg_q, 4)})
            all_rows.append(rows[-1])
            
        plt.plot([r["cost_usd"] for r in rows], [r["quality"] for r in rows], 
                 marker=markers[idx % len(markers)], color=colors[idx % len(colors)], 
                 label=name, linewidth=2)
                 
    plt.title("Cost-Quality Frontier: Router Evaluation")
    plt.xlabel("Cost (USD, estimated input tokens, cache-aware)")
    plt.ylabel("Execution Health Proxy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "frontier.png")
    
    with open(out_dir / "frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "adopt_frac", "cost_usd", "quality"])
        w.writeheader()
        w.writerows(all_rows)
        
    print(f"Successfully generated results/frontier.png and results/frontier.csv")

if __name__ == "__main__":
    main()