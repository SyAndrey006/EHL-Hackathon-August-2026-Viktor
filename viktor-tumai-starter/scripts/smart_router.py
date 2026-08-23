#!/usr/bin/env python3
"""Opening-call commitment router: one gate, one model, zero switches."""
import argparse, csv, json, math, statistics, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from load_trajectories import group_trajectories, iter_requests
from cost_model import load_pricing, logged_route, trajectory_cost
from outcome_evaluator import configure_estimator, estimate_route_quality

CHEAP={"claude":"claude-sonnet-5","gpt":"gpt-5.6-luna"}
def cheap_for(m): return CHEAP["claude"] if m.startswith("claude") else CHEAP["gpt"]
def sigmoid(x): return 1/(1+math.exp(-max(-30,min(30,x))))
def logit(p): p=max(1e-6,min(1-1e-6,p)); return math.log(p/(1-p))


class Classifier:
    """L2 logistic model; scaling multipliers favor the specified early signals."""
    nums=("has_reasoning","has_input_image","position_fraction")
    def fit(self,rows):
        train=[r for r in rows if r["outcome_observed"]=="1" and r["proxy_simple"]!=""]
        self.cats=sorted({r["task_category"] for r in rows}); self.prior=statistics.fmean(int(r["proxy_simple"]) for r in train)
        self.mean={k:statistics.fmean(float(r[k]) for r in train) for k in self.nums}; self.sd={k:max(.1,statistics.pstdev(float(r[k]) for r in train)) for k in self.nums}
        self.mult=[1,2.5,2.5,2,*([1.75]*len(self.cats))]; self.w=[0.0]*len(self.mult); self.w[0]=logit(self.prior)
        for _ in range(1000):
            g=[0.0]*len(self.w)
            for r in train:
                x=self.vector(r); e=sigmoid(sum(a*b for a,b in zip(self.w,x)))-int(r["proxy_simple"])
                for j,v in enumerate(x): g[j]+=e*v
            for j in range(len(self.w)): self.w[j]-=.08*(g[j]/len(train)+(0 if j==0 else .08*self.w[j]))
        return self
    def vector(self,r):
        x=[1,*[(float(r[k])-self.mean[k])/self.sd[k] for k in self.nums],*[int(r["task_category"]==c) for c in self.cats]]
        return [a*b for a,b in zip(x,self.mult)]
    def predict(self,r): return sigmoid(sum(a*b for a,b in zip(self.w,self.vector(r))))


def opening_score(tid,calls,row,clf):
    cheap=cheap_for(calls[0]["model"]); matched=estimate_route_quality(tid,[cheap]*len(calls)); proxy=clf.predict(row)
    return sigmoid(logit(matched)+.75*(logit(proxy)-logit(clf.prior)))


def route_trajectory(tid,calls,feature_rows,clf,threshold):
    """Evaluate only call_index=1 and commit the chosen model for every call."""
    opener=next(r for r in feature_rows if int(r["call_index"])==1); score=opening_score(tid,calls,opener,clf)
    logged=calls[0]["model"]; chosen=cheap_for(logged) if score>=threshold else logged
    return [chosen]*len(calls),score


def main():
    p=argparse.ArgumentParser(); p.add_argument("export",nargs="?",default="export"); p.add_argument("features",nargs="?",default="results/call_features.csv"); p.add_argument("--outcomes",default="results/outcomes.json"); p.add_argument("--output",default="results/routes_smart.jsonl"); p.add_argument("--threshold",type=float,default=.85); a=p.parse_args()
    with open(a.features,newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    clf=Classifier().fit(rows); configure_estimator(a.outcomes,a.features); by={}
    for r in rows: by.setdefault(r["trajectory"],[]).append(r)
    groups=group_trajectories(r for _,_,r in iter_requests(a.export)); pricing=load_pricing(); records=[]
    for tid,calls in groups.items():
        route,score=route_trajectory(tid,calls,by[tid],clf,a.threshold); cl,_=trajectory_cost(calls,logged_route(calls),pricing); cr,_=trajectory_cost(calls,route,pricing)
        records.append({"trajectory":tid,"n_calls":len(calls),"logged_model":calls[0]["model"],"route":route,"cost_logged_usd":round(cl,6),"cost_routed_usd":round(cr,6),"switches":sum(route[i]!=route[i-1] for i in range(1,len(route)))})
    assert all(r["switches"]==0 and len(set(r["route"]))==1 for r in records)
    with open(a.output,"w",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(r)+"\n")
    logged=sum(r["cost_logged_usd"] for r in records); routed=sum(r["cost_routed_usd"] for r in records); cheap=sum(r["route"][0]!=r["logged_model"] for r in records)
    print(f"threshold={a.threshold:.2f} committed_cheap={cheap}/{len(records)} switches=0")
    print(f"logged=${logged:.4f} smart=${routed:.4f} savings=${logged-routed:.4f} ({1-routed/logged:.1%})"); print(f"wrote {a.output}")
if __name__=="__main__":main()
