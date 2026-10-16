# Changelog

Four versions of the model. Each revision came out of review finding that the previous version was measuring something other than what it claimed, or feeding it with assumed numbers, so this file records the reasoning rather than only the diffs.

## v4 (2026-10-08, current)

The version in `gating_sim.py`, behind the short-paper revision. v3 is archived as `archive/gating_sim_v3.py` with its outputs.

### Why

The IPCCC reviews made two points that v3 could not answer from inside: the inputs were assumed rather than measured, and a model whose windows are exponential by construction confirms its own assumption. v4 keeps the v3 semantics and replaces the inputs.

### Changed

- **Units.** Update size in GB, rates in Mbps, time in minutes. The delivery-minute unit was opaque to a reviewer and hid the rate assumption.
- **Windows.** Three sources: the v3 exponential windows (kept for comparison), bootstrap resampling from measured sets (DFR dock windows per agency under the minimum fleet consistent with the overlaps; Chicago taxi gaps), and a deterministic replay of real device schedules from a stated campaign start time. The first window's residual is the renewal-theory residual in every source.
- **Rates.** Anchors from Ookla 2026 Q2 tiles (fixed 372 Mbps, Chicago mobile 297, rural mobile 137, mobile 10th percentile 28) and the documented Starlink median (118). The measured fixed-to-mobile ratio is 1.3 to 2.4, not the 3.6 and 13 that v3 assumed.
- **Install time.** 20 min (dock, documented) and 30 min (vehicle, documented) instead of 5 min, and swept.
- **Metric axis.** At measured rates a multi-gigabyte transfer takes minutes, so completion against size at a six-hour deadline is flat for most settings. The completion-before-redeployment rate is now also reported against the deadline, summarised by T95 (earliest deadline at which 95 percent of the fleet is current). S95 replaces the margin-based crossover: the largest size that still meets the target, no margin to choose.

### Added

- **Reach.** A depot window is one spent where local staging exists. Docked drones: every window. AVs: gaps of at least 120 min by default, or a swept probability. This is the reach-versus-rate trade: cellular or satellite push reaches every window; staging has the rate only where it exists.
- **Hybrid architecture.** Best available link per window.
- **Two baselines from the review.** A window-aware scheduler that lets transfer span windows (install still atomic), and a bandwidth-only selector. The run reports where each one says centralized push meets the target while atomic completion does not.
- **Agency spread and start-hour replay.** Population sensitivity across five agencies; deterministic diurnal runs.
- **Contact rate.** The AV gossip rate β is measured from the Cabspotting taxi traces (536 cabs, 24 days): 0.82 contacts per node per minute at 100 m range in daytime when every contact transmits, two orders of magnitude above the v3 assumption of 0.006. The median contact lasts 60 s, so the sweep also carries the duration-gated rates (only contacts of at least 1, 2, 5, 10, 20 min: 0.52 down to 0.013). No contact data exists for docked drone fleets; the drone value stays the v3 assumption and is labelled as such.
- **Peer links and the contact gate (2026-10-09).** The gossip rate assumes transmission on every contact. The run now also reports, for documented peer-link rates, the contact length a whole-image transfer needs and the share of measured Cabspotting contacts that long, and runs gossip at that gated rate for the 0.42 GB firmware anchor and the 4 GB headline. Rates are documented, never assumed: IEEE 802.11p physical-layer rates (6 Mbps default, 27 Mbps top, 10 MHz channel) and the field-measured LTE-V2X PC5 throughput with commercial radios (0.72 Mbps mean, about 2.6 Mbps peak per pair, four nodes on 20 MHz) for ground vehicles; vendor "up to" rates of MANET mesh radios flown on UAVs (100 and 80 Mbps) and a research SDR mesh demonstration (99.84 Mbps per link) for drones. No 5G NR sidelink field figure exists, so none is used. A transfer-time table per link type for both classes is printed with the run. Finding: at 4 GB only the 802.11p top rate has any measured contacts long enough (1.6 percent, gated rate 0.013 per node per minute, the order of the v3 assumption); at 0.42 GB a fifth of contacts suffice at 27 Mbps.
- **Exact engines.** For ρ = 0 completion is evaluated on each device's window sequence without time stepping, so the full sweep set runs in minutes. The resumability sweep uses a second exact engine that walks the windows carrying ρ times the transfer progress; it reproduces the main engine at ρ = 0 and the non-atomic baseline at ρ = 1, and the run prints both differences. The v3 stepped engine stays only as an independent cross-check.
- **Paper figures (2026-10-15).** `scripts/make_paper_figures.py` renders the manuscript figures from `expected/results_v4.csv` and `expected/sweeps_v4.csv` only, at IEEE column width with print-sized type and the Okabe-Ito colourblind-safe palette (architecture by colour and marker, device class by line style). `gating_sim.py` and the PNGs in `figures/` are unchanged; the paper figures trace to the byte-compared data rather than to a plot the verifier does not check.
- **Tests.** `tests/test_gating_sim.py`: every architecture and baseline against a per-device reference implementation, ordering invariants, monotonicity, the engines against each other, a hand-computed replay, the summaries, the seeds. `python gating_sim.py --quick` runs the whole pipeline with a few replications.

### Fixed (2026-10-09, before the reference run)

- **Depot rule on the window in progress.** The rule that calls a stationary window a depot stop (AVs: at least 120 min) was applied to the remaining part of the window under way at release rather than to its full length, so a long stop already in progress was often classed as a curb stop. Both engines now apply the rule to the full length; the replay already did.
- **Hybrid at a depot.** Takes the faster of the local and backhaul links rather than the local link unconditionally. No sweep in this version has a backhaul faster than the local link, so the numbers do not move; the definition now matches the text.
- **Resumability sweep.** Moved from the one-minute stepped engine to the exact engine. The stepped engine credits whole steps and counts the transfer-completion step towards the install, so it runs about two minutes optimistic per window; that bias is now stated and checked instead of being inside a reported number.
- **Zero-length windows.** 764 zero-length taxi records (gaps of zero between adjoining trips, trips with no duration) are dropped before resampling; they were never windows.
- **Renewal check.** The v3 closed form is a long-horizon approximation (it ignores the window under way at release and the deadline cut). The run now reports its deviation at six hours and beyond two cycles separately from the overall maximum, which is dominated by the first cycle.

### Fixed (2026-10-09, recall study set; the simulator and the reference run are not affected)

- **Recall classifier matches whole words.** `scripts/derive_nhtsa_ota.py` matched the manufacturer-name lists as substrings, which put Zero Motorcycles in the operator class ("TORC" inside "MOTORCYCLES") and Polestar and Rivian in the third-party fleet class ("MOTIV" inside "AUTOMOTIVE"). The over-the-air pattern is bounded on both ends; "cover the air gap" (a trim-piece recall) had passed. Study set 177 to 176, operator class 16 to 15, fleet class 16 to 6; no per-campaign completion figure changed. New columns `models`, `n_models` and `dual_path` (the remedy text also names a dealer visit), and a remote-only per-class summary in `nhtsa_summary.json`.

### Effect on results

Reference run frozen 2026-10-09 (`expected/run_v4.log`, R = 60, N = 500). T95 is the earliest deadline, on a 10-minute grid, at which 95 percent of the fleet is current with a 4 GB update.

- **Measured windows against the v3 assumption.** Drones (Cincinnati docks, 20-min install): push reaches the target at 60 min on bootstrap windows and at 60 min in replay, against 160 min under the v3 exponential windows. AVs (Chicago taxi, 30-min install): 160 min bootstrap, 150 min replay, and never inside six hours under the exponential windows (0.82 at six hours). The exponential assumption was pessimistic for both classes; bootstrap and replay agree to within one grid step.
- **Size no longer separates the architectures.** At measured rates every architecture that reaches the target at all does so for every size in the 0.25 to 10 GB sweep (S95 = 10, the top of the sweep). The v3 margin crossover does not occur. The install gate and the window length set T95; the link rate moves it by one grid step between rural cellular, satellite and metro cellular.
- **Install time is the dominant driver.** Push T95 runs from 20 to 170 min for drones and 40 to 310 min for AVs as the install goes from 5 to 60 min.
- **Reach versus rate.** AV depot-only staging never reaches the target unless every stop has staging, because only 19 percent of daytime stops are long ones. At the metro cellular rate the hybrid gains nothing from reach (160 to 150 min); at the poor-cellular stress rate (28 Mbps) the hybrid falls from 240 to 150 min as reach goes from none to all stops. Push alone is within the best architecture's T95 by one grid step at every rate of 118 Mbps or more; at 5 Mbps it never reaches the target for either class.
- **Gossip at the measured contact rate** is not the bottleneck: with every contact transmitting it reaches the target at the same deadline as push (50 min drones, 150 min AVs), because the install gate still binds. It needs contacts of at least 5 min to stay within about 30 min of push, and never reaches the target inside six hours at the v3 assumed rates or at the 20-min gate.
- **Peer links at documented rates.** With the contact gate, no documented vehicle-to-vehicle link carries the 4 GB headline image inside measured contacts: the 802.11p top rate needs a 19.8 min contact and 1.6 percent of contacts are that long (gated rate 0.013 per node per minute, completion 0.75 at six hours, target not reached); the 6 Mbps default and the measured LTE-V2X rates need longer than any binned contact. At the 0.42 GB firmware package the peer path works: 802.11p at 27 Mbps reaches the target at 160 min (a fifth of contacts suffice), at 6 Mbps at 220 min; measured LTE-V2X peaks reach 0.63. The transfer-time table shows the other side of the same coin: at the stop links a 4 GB transfer takes 0.5 to 4.5 min for drones and 1.4 to 4.5 min for AVs, and 96 to 97 percent of measured dock windows but only 42 to 45 percent of measured taxi stops are long enough for transfer plus install.
- **Baselines.** The window-aware non-atomic scheduler is one grid step optimistic (50 vs 60 min drones, 150 vs 160 min AVs); the bandwidth-only selector promises 40 min for drones and 90 min for AVs. Resumable transfer buys one grid step at most (ρ from 0 to 1: 60 to 50 min, 160 to 150 min), because transfer takes minutes at measured rates.
- **Population and diurnal spread.** Across the five agencies push T95 runs 30 to 60 min. AV replay by campaign start hour: 150 min at 08:00 and 12:00, 190 min at 16:00, 100 min at 20:00; over active device-days only, 140 to 220 min.
- **Checks.** The two exact engines agree to zero; the v3 stepped engine is within 0.012 of the exact one; the renewal closed form is within 0.011 at six hours and off by up to 0.22 inside the first cycle.

## v3 (2026-07-20)

Archived as `archive/gating_sim_v3.py` with `archive/results_v3.csv`, `archive/run_v3.log`, and the v3 figures. The version behind the first submission.

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
