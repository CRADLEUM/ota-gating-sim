# Archive

Superseded material, kept for provenance. Do not run these or cite their numbers.

| File | What it is |
| --- | --- |
| `gating_sim_v2.py` | The v2 model. Correct within-window semantics, but the initial residual window was drawn as `U(0,1) × Exp(mean)` instead of `Exp(mean)`. |
| `results_v2.csv` | v2 results. Superseded by `../expected/results_v3.csv`. |
| `run_v2.log` | v2 stdout. |
| `crossover_v2.png` | v2 main figure. |
| `drivers_v2.png` | v2 two-panel drivers figure, before the resumability panel was added. |

The current model is `../gating_sim.py`. `../CHANGELOG.md` explains what changed at each step and why.

v1 is not archived here. It reduced completion to an aggregate bandwidth condition, which made the gate non-binding, so its outputs do not measure what the surrounding text claimed. The reasoning is recorded in the changelog instead.
