#!/usr/bin/env python3
"""Logged, baseline, and opening-gated frontiers with uncertainty/sensitivity."""
import argparse,csv,json,sys,struct,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from outcome_evaluator import configure_estimator,estimate_route_quality,wilson_interval,gamma_to_flip,cem_diagnostics

def load(p):
    with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]

def curve(records,name):
    enriched=[]
    for r in records:
        logged=estimate_route_quality(r["trajectory"],[r["logged_model"]]*r["n_calls"]); routed=estimate_route_quality(r["trajectory"],r["route"])
        saving=r["cost_logged_usd"]-r["cost_routed_usd"]; enriched.append((max(0,logged-routed),-saving,r,logged,routed))
    enriched.sort(key=lambda x:(x[0],x[1])); rows=[]
    for frac in [i/10 for i in range(11)]:
        adopted={id(x[2]) for x in enriched[:round(frac*len(enriched))]}; probs=[];cost=0
        for _,_,r,lq,rq in enriched:
            use=id(r) in adopted; probs.append(rq if use else lq); cost+=r["cost_routed_usd"] if use else r["cost_logged_usd"]
        q=sum(probs)/len(probs);lo,hi=wilson_interval(sum(probs),len(probs))
        rows.append({"policy":name,"adopt_frac":frac,"cost_usd":round(cost,4),"expected_success":round(q,4),"ci95_low":round(lo,4),"ci95_high":round(hi,4)})
    return rows

def fallback_png(path,logged,base,smart):
    """Dependency-free PNG fallback so the final artifact is always regenerated."""
    w,h=700,450;pix=[bytearray([255,255,255]*w) for _ in range(h)]
    def dot(x,y,color,r=3):
        for yy in range(max(0,y-r),min(h,y+r+1)):
            for xx in range(max(0,x-r),min(w,x+r+1)):pix[yy][3*xx:3*xx+3]=bytes(color)
    allr=[logged]+base+smart; xmin=min(r['cost_usd'] for r in allr);xmax=max(r['cost_usd'] for r in allr);ymin=min(r['expected_success'] for r in allr)-.02;ymax=1.01
    def xy(r):return 60+int(600*(r['cost_usd']-xmin)/max(1e-9,xmax-xmin)),400-int(350*(r['expected_success']-ymin)/max(1e-9,ymax-ymin))
    def line(a,b,c):
        x0,y0=a;x1,y1=b;n=max(abs(x1-x0),abs(y1-y0),1)
        for i in range(n+1):dot(round(x0+(x1-x0)*i/n),round(y0+(y1-y0)*i/n),c,1)
    line((60,400),(660,400),(0,0,0));line((60,400),(60,50),(0,0,0))
    for data,color in ((base,(140,140,140)),(smart,(103,72,253))):
        points=[xy(r) for r in data]
        for a,b in zip(points,points[1:]):line(a,b,color)
        for p in points:dot(*p,color,3)
    dot(*xy(logged),(0,0,0),5)
    raw=b''.join(b'\x00'+bytes(row) for row in pix)
    def chunk(t,d):return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b''))

def main():
    p=argparse.ArgumentParser();p.add_argument("smart",nargs="?",default="results/routes_smart.jsonl");p.add_argument("--baseline",default="results/routes.jsonl");p.add_argument("--outcomes",default="results/outcomes.json");p.add_argument("--features",default="results/call_features.csv");p.add_argument("--output",default="results/frontier_2.csv");a=p.parse_args()
    configure_estimator(a.outcomes,a.features);base=curve(load(a.baseline),"baseline");smart=curve(load(a.smart),"smart"); logged=dict(base[0]);logged.update(policy="logged",adopt_frac=0.0)
    for b,s in zip(base,smart):s["gamma_to_flip_vs_baseline"]=round(gamma_to_flip(s["expected_success"],b["expected_success"]),3);b["gamma_to_flip_vs_baseline"]=1.0
    logged["gamma_to_flip_vs_baseline"]=1.0;rows=[logged]+base+smart; fields=list(rows[-1]);
    with open(a.output,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    bl,sm=base[-1],smart[-1];lc=logged["cost_usd"]
    print(f"logged cost=${lc:.4f}");print(f"baseline savings=${lc-bl['cost_usd']:.4f} ({1-bl['cost_usd']/lc:.1%}) quality={bl['expected_success']:.3f} CI[{bl['ci95_low']:.3f},{bl['ci95_high']:.3f}]")
    print(f"smart savings=${lc-sm['cost_usd']:.4f} ({1-sm['cost_usd']/lc:.1%}) quality={sm['expected_success']:.3f} CI[{sm['ci95_low']:.3f},{sm['ci95_high']:.3f}] retention_vs_baseline={sm['expected_success']/bl['expected_success']:.1%}")
    print(f"smart full-policy strictly up-left baseline: {sm['cost_usd']<bl['cost_usd'] and sm['expected_success']>bl['expected_success']}")
    print(f"CEM diagnostics: {cem_diagnostics()}");print("FAILURE MODES: sample sparsity; observational confounding; terminal-label noise; model-based fractional Wilson CI.");print(f"wrote {a.output}")
    try:
        import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
        plt.figure(figsize=(7,4.5));plt.scatter([logged['cost_usd']],[logged['expected_success']],label='logged',color='black',s=60)
        for name,data,color in [('baseline',base,'#999999'),('smart',smart,'#6748FD')]:plt.plot([r['cost_usd'] for r in data],[r['expected_success'] for r in data],'o-',label=name,color=color)
        plt.xlabel('cost (USD, cache-aware estimated input)');plt.ylabel('expected success');plt.title('Cost-quality frontier (95% intervals in CSV)');plt.legend();plt.tight_layout();png=Path(a.output).with_suffix('.png');plt.savefig(png,dpi=160);print(f"wrote {png}")
    except ImportError:
        png=Path(a.output).with_suffix('.png');fallback_png(png,logged,base,smart);print(f"wrote {png} (standard-library fallback)")
if __name__=="__main__":main()
