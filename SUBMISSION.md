# Viktor Smart Router — Viktor Challenge submission

Team: Viktor Smart Router team (participant identity is recorded in the hackathon portal)

Objective: Reduce estimated agent input cost without paying model-switch cache-reset penalties.

Routing signal: An Opening-Call Commitment gate combines an L2-regularized classifier over the first call (`has_reasoning`, `has_input_image`, position, and task category) with a CEM-derived cheap-model success prior. At threshold 0.75 it commits the complete trajectory either to the cheaper sibling model or to the logged model. It never changes model inside a trajectory.

Headline result: Estimated input-side cost falls from $0.2524 to $0.1126 (−55.4%) with expected task success 0.833. The policy reroutes 15 of 25 reconstructed trajectories and records zero mid-trajectory switches.

Off-policy method: Terminal task-completion detection plus Coarsened Exact Matching on `(task_category, has_input_image, length_bucket)`, Wilson intervals, and a Rosenbaum-style sensitivity ratio.

Named weakness: This is an observational estimate, not an independent held-out or online test. Weighted ESS is 13.43, the Smart estimate has a wide 95% interval `[0.646, 0.932]`, and approximately `Γ = 1.10` unobserved selection bias could erase its advantage over the Viktor length heuristic.

Cost scope: Token counts are estimated as serialized characters divided by four. Reported dollars cover input tokens only. Consecutive same-model prefixes receive the modeled cache-read price; output cost, cache writes, and provider usage counters are unavailable.

Reproduction commands are provided in `README.md`. The challenge dataset is intentionally excluded because its license prohibits redistribution.
