# Changelog

Three versions of the model. Both revisions came out of review finding that the previous version was measuring something other than what it claimed, so this file records the reasoning rather than only the diffs.

## v3 (2026-07-20) — current

The version in `gating_sim.py`, and the one behind the submitted results.

### Fixed

**Initial residual window length.** The time remaining in a device's first window was drawn as `U(0,1) × Exp(mean)`, which has mean `m/2`. For exponential windows the stationary residual is itself `Exp(mean)`, by memorylessness. The old draw contradicted the memoryless model the paper invokes and started every device part-way through a window that was too short on average.

Effect on results: the drone crossover is unchanged at S\*=25. The AV crossover moves from 55 to 65. The top of the backhaul sweep moves from 105 to 115. Maximum standard deviation across configurations is 0.023.

The drone result being unchanged is not a coincidence. Drone windows are short enough that the first partial window rarely decides anything; AV windows are long enough that it does.

### Added

- **Resumability sweep.** A ρ parameter carrying a fraction of transfer progress across a window boundary, swept from 0 to 1. This exists because the atomicity assumption is the load-bearing one, and a reader is entitled to see how much of the effect survives without it. Install progress stays atomic at every ρ.
- **Gossip contact-rate sweep.** Completion against β at fixed update size. Gossip performs poorly under the assumed drone contact rate, and a reviewer is right to ask whether that is a modeling choice rather than a finding. The sweep shows where gossip becomes competitive instead of leaving it as a strawman.
- **Closed-form check.** A renewal approximation for atomic within-window completion, compared against the simulated means. Agreement to within 0.080 everywhere. This validates the simulator against an independently derived expression; it is not a second result.
- **Merged main figure.** Both device classes on one axis, topology by colour and class by line style, since the contrast between the two classes is the point being made.

Prior artifacts are in `archive/`.

## v2 (2026-07-06)

Archived as `archive/gating_sim_v2.py`, with `archive/results_v2.csv` and `archive/run_v2.log`.

### Fixed

**The install gate did no work.** In v1, received bytes accumulated across stationary windows. Completion therefore reduced to `throughput × H ≥ S`, which is a pure bandwidth condition, and the gate was non-binding. The apparent difference between device classes was produced entirely by the backhaul-to-local rate ratio, not by window length. A stage-3 review pass, methods reviewer plus devil's advocate, identified this correctly.

This mattered because the paper's claim is that gating is an independent driver. Under v1 the simulation could not have supported that claim, whatever its output happened to show.

**The fix.** Model deploy-on-complete as an atomic maintenance session. A device becomes current only when a single stationary window is long enough to transfer the whole update and complete the install, contiguously, with no cross-window carryover. Under this semantics a slow backhaul can deliver enough total bytes over the horizon and still never deliver enough within one window, which is what makes window length load-bearing.

### Added

**Gate-isolation experiment.** Hold every transfer rate fixed and vary only the stationary-window length. If the preferred topology changes, gating is an independent driver. This is the experiment that answers the v1 objection directly, and it is the reason the figure exists.

## v1

Not shipped in this repository, deliberately.

The v1 model reduced to a bandwidth condition, as described above. Its outputs are not wrong arithmetic, but they do not measure what the surrounding text said they measured, and publishing them alongside the current results would invite exactly the confusion the v2 revision resolved. The reasoning is preserved here instead.
