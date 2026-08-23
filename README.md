# Viktor Hackathon — Cache-Aware LLM Router

This repository contains our Viktor Hackathon solution for routing multi-step agent
trajectories between premium and lower-cost language models.

The final policy is an **opening-call commitment router**. It evaluates the first call,
selects either the cheap sibling model or the logged model, and keeps that model for the
entire task. This guarantees zero mid-trajectory switches and preserves the provider's
shared-prefix cache discount.

## Headline result

On the included synthetic challenge-shaped export, threshold `0.75` produced:

- **55.4% estimated input-cost savings** versus the logged policy
- **0 model switches** across 25 reconstructed trajectories
- matched expected success of **0.833**, versus **0.820** for the baseline
- Coarsened Exact Matching overlap in 5/6 strata, with weighted ESS of 13.43

These are offline estimates, not measured production outcomes. Token usage is estimated
from serialized text, and counterfactual quality is based on observational matching.

## Repository layout

The runnable project is in [`viktor-tumai-starter/`](./viktor-tumai-starter/).

| Path | Purpose |
|---|---|
| `scripts/smart_router.py` | Opening-call classifier and zero-switch routing policy |
| `scripts/feature_analyzer.py` | Extracts call-level routing features |
| `scripts/outcome_evaluator.py` | Completion detection, CEM matching, uncertainty, and cache checks |
| `scripts/cost_model.py` | Cache-aware input-cost repricing |
| `scripts/baseline_router.py` | Small-trajectory baseline heuristic |
| `scripts/plot_frontier_2.py` | Logged/baseline/smart frontier comparison |
| `export/` | Challenge export; intentionally not committed |
| `results/` | Routes, estimates, reports, and graphs |

## Requirements

- Python 3.10+
- No GPU or API key
- Standard library only for the main pipeline
- Matplotlib is optional; graph generation has a dependency-free PNG fallback

On Windows, the project includes a local Python runtime. Use `.\.python\python.exe` in
place of `python` if Python is not installed globally.

## Quick start

From the Git repository root:

```powershell
cd viktor-tumai-starter
```

If no challenge export is available, generate the deterministic synthetic sample:

```powershell
python scripts/make_synthetic_sample.py
```

For real challenge data, place extracted `trajectories_v1_*.jsonl` files in `export/`.
The dataset is challenge-use-only and must not be committed or redistributed.

## Run the complete pipeline

```powershell
# 1. Validate the export and reconstruct trajectories
python scripts/load_trajectories.py export/

# 2. Extract predictive features
python scripts/feature_analyzer.py export/

# 3. Detect outcomes, build CEM estimates, and verify cache repricing
python scripts/outcome_evaluator.py export/ results/call_features.csv

# 4. Generate baseline routes
python scripts/baseline_router.py export/

# 5. Run the final opening-call commitment router
python scripts/smart_router.py export/ results/call_features.csv --threshold 0.75

# 6. Generate the comparison report and graph
python scripts/plot_frontier_2.py results/routes_smart.jsonl --baseline results/routes.jsonl --output results/frontier_2.csv
```

Example with the bundled Windows runtime:

```powershell
.\.python\python.exe scripts\smart_router.py export\ results\call_features.csv --threshold 0.75
```

## Main outputs

| Output | Description |
|---|---|
| `results/call_features.csv` | Call features and observable action-complexity proxy |
| `results/outcomes.json` | Logged model and detected completion per trajectory |
| `results/routes.jsonl` | Baseline decisions and cache-aware costs |
| `results/routes_smart.jsonl` | Smart decisions; every route has zero switches |
| `results/frontier_2.csv` | Costs, expected success, Wilson intervals, and sensitivity values |
| `results/frontier_2.png` | Logged versus baseline versus smart frontier graph |

## How the router works

1. Requests are reconstructed into trajectories using their opening task text.
2. A lightweight logistic classifier learns from the action-shape proxy in
   `call_features.csv`.
3. It emphasizes reasoning, image modality, position, and task category.
4. Its prediction is calibrated with the matched cheap-model success estimate.
5. At `call_index == 1`, it commits the complete trajectory to either:
   - `gpt-5.6-luna` / `claude-sonnet-5`, or
   - the logged model.
6. The model never changes after that decision.

## Evaluation and cache accounting

Quality is estimated with Coarsened Exact Matching over:

```text
(task_category, has_input_image, trajectory_length_bucket)
```

The evaluator reports matched-strata overlap, effective sample size, Wilson confidence
intervals, and a Γ odds-ratio sensitivity value for unobserved confounding.

Costs are input-side estimates. The first prompt is charged at the uncached rate. For
later calls on the same model, the shared prefix receives cache-read pricing while newly
appended tokens remain uncached. The evaluator independently verifies this calculation.

## Limitations

- The export contains no true quality labels or measured token usage.
- Completion labels inferred from terminal messages may be noisy.
- Observational matching cannot eliminate hidden assignment confounding.
- Treatment overlap is sparse, producing wide confidence intervals.
- `proxy_simple` describes logged action shape, not guaranteed cheap-model correctness.
- No tested threshold strictly beat the baseline on both cost and estimated quality;
  `0.75` is the selected trade-off point, not a universal optimum.

## Reproducibility checklist

Before presenting or submitting, rerun the pipeline and confirm:

- every record in `routes_smart.jsonl` has `switches: 0`
- the cache repricing sanity check prints `PASS`
- `frontier_2.csv` and `frontier_2.png` were regenerated
- quoted costs are clearly described as estimated input-side costs
