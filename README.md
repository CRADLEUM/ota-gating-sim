# ota-gating-sim

Simulation code and artifacts for the quantitative section of a paper on over-the-air update delivery for autonomous devices that operate in public space.

One file, `gating_sim.py`, produces every number in that section. `scripts/make_paper_figures.py` renders the paper's three figures from the frozen outputs in `expected/`.

**Status:** accepted as a short paper at IPCCC 2026; the v4 results here feed the revised manuscript. Nothing here is peer reviewed yet.

## What the model does

Devices alternate between two states: operating in public space, where they cannot install an update, and stationary at a dock or curb, where they can. In v4 the window durations come from measured data (police drone flight logs, Chicago taxi trips) by bootstrap resampling or by deterministic replay of real schedules; the v3 exponential windows are kept as a comparison run.

A device counts as current only when a single stationary window is long enough to transfer the whole update at its in-window rate and complete an atomic install, contiguously. Partial progress is lost when the window ends.

That semantics is the point of the model. Over a fixed patch deadline, a slow link can move plenty of total bytes and still never move enough inside one window. When that happens the window length sets the answer and the link rate does not.

`docs/model.md` has the state machine, the parameter table, the metric, and the assumptions.

## Requirements

Python 3.9 or later, numpy, and matplotlib.

```
pip install -r requirements.txt
```

Verified reproducing on Python 3.12.10, numpy 2.4.6, matplotlib 3.11.0.

## Run it

```
python gating_sim.py
```

Takes about five minutes. Reads `data/sim_inputs/` and writes into its own directory:

| Output | Contents |
| --- | --- |
| `results_v4.csv` | Completion rate per class, window source, architecture, update size, and deadline (mean and std over replications) |
| `sweeps_v4.csv` | T95 and completion at six hours for every sweep |
| `replay_v4.csv` | Deterministic replay curves per campaign start hour, all device-days and active device-days |
| `crossover.png` | Completion against deadline at a 4 GB update, both classes, four architectures |
| `drivers.png` | Four panels: install time, backhaul rate, gate isolation, reach |
| `sources.png` | Exponential versus bootstrap versus replay windows, centralized push |
| stdout | Transfer time by link type for both classes, summary tables, T95 and S95 per source, baseline disagreement, nine sweeps (including gossip at documented peer-link rates through the measured contact gate), engine checks |

### Data pipeline

| Step | Script | Writes |
| --- | --- | --- |
| pull raw data | `scripts/pull_data.py` | `data/raw/` (ignored) and `data/raw/manifest.json` (URL, bytes, SHA-256, pull time) |
| derive windows and rates | `scripts/derive_dfr_windows.py`, `derive_chicago_windows.py`, `derive_ookla_rates.py`, `derive_cabspotting_contacts.py`, `derive_nhtsa_ota.py` | `data/derived/` |
| compact sim inputs | `scripts/make_sim_inputs.py` | `data/sim_inputs/` (committed) |
| evidence file for the paper | `scripts/report_measured_inputs.py` | the paper repository's `03_evidence/MEASURED_INPUTS.md` |
| paper figures from the frozen run | `scripts/make_paper_figures.py` | `fig1_t95`, `fig2_drivers`, `fig3_sources` (PDF and PNG) in the directory given by `--outdir`, read from `expected/` |

Raw files are not redistributed; the manifest lets a reader confirm a fresh pull matches by hash.

Reference copies are committed under `expected/` and `figures/`.

## Check that it reproduces

```
python -m pytest tests/        # unit tests, about 5 s
python gating_sim.py --quick   # whole pipeline with a few replications, a few minutes
python verify.py               # full reference run, about five minutes
```

The tests check every architecture and baseline against a plain per-device reference implementation, the ordering invariants between them, the two exact engines against each other, the v3 stepped engine within its stated bias, a hand-computed replay, and the seeds. `docs/model.md` lists them.

This copies the model to a scratch directory, runs it, and compares both the stdout log and the CSV against the reference artifacts. It exits non-zero on any difference, and leaves the working tree clean.

The run is deterministic. A fixed seed feeds per-replication seeds derived arithmetically rather than through `hash()`, which Python randomizes per process and which would otherwise have made runs vary.

Figures are not byte-compared, because matplotlib output shifts between versions and a pixel difference is not a reproduction failure. The CSV carries the data.

## Scope and limits

Worth reading before citing any number from this repository.

- **It is not a measurement.** All three delivery architectures are constructed baselines parameterized from a table, not deployments. The output compares them under a stated model. It does not predict absolute completion rates for any real fleet.
- **The atomicity assumption drives the result.** A fully resumable download regime reduces completion to an aggregate bandwidth condition. The effect studied here is specific to the atomic install regime, which is why resumability is swept rather than assumed away.
- **The windows are measured, the fleets are not the reader's.** Drone windows come from five police drone-as-first-responder programs under the smallest fleet consistent with the flight overlaps (a lower bound on dock time); AV windows come from Chicago taxi trips, with start times rounded to 15 minutes in the public data. The install times are documented vendor figures, swept. The exponential windows of v3 are kept only as a comparison run.
- **Gossip is a logistic epidemic model.** The AV contact rate is measured from the Cabspotting taxi traces with transmission assumed on every contact, which is an upper bound; the sweep gates it by contact length. The drone contact rate is still an assumption, labelled as such, because no contact data exists for docked fleets.

## Version history

The model was revised twice for correctness, both times after review found the earlier version was measuring the wrong thing. `CHANGELOG.md` records what changed and why. `archive/` holds the superseded version with its artifacts.

## Citation

See `CITATION.cff`. Update it on acceptance.

## License

Code and documentation: MIT, see `LICENSE`.

Ookla data: `data/derived/ookla/ookla_rates.csv` and `ookla_rates.json` are adapted from Speedtest® by Ookla® Global Fixed and Mobile Network Performance Maps (Ookla, 2026 Q2 fixed and mobile tiles, https://registry.opendata.aws/speedtest-global-performance/, accessed 8 Oct. 2026). The values are the authors' analysis: per-region tile and test counts, and medians and 10th percentiles of the per-tile mean speeds. These two files are shared under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/), not MIT, so reuse must be non-commercial and keep the same license. The rate anchors in `gating_sim.py` and `docs/model.md` are read from them. Ookla trademarks used under license and reprinted with permission.
