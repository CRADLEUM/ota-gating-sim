# ota-gating-sim

Simulation code and artifacts for the quantitative section of a paper on over-the-air update delivery for autonomous devices that operate in public space.

One file, `gating_sim.py`, produces every number and both figures in that section.

**Status:** the paper is under review at IPCCC 2026. Nothing here is peer reviewed yet.

## What the model does

Devices alternate between two states: operating in public space, where they cannot install an update, and stationary at a dock or curb, where they can. Window durations are exponential.

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

Takes a few minutes. Writes into its own directory:

| Output | Contents |
| --- | --- |
| `results_v3.csv` | Mean and standard deviation of the completion rate per class, topology, and update size |
| `crossover.png` | Completion against update size for both device classes, with 95 percent bands |
| `drivers.png` | Three panels: gate isolation, backhaul sensitivity, resumability |
| stdout | Summary tables, crossover points, and four sweeps |

Reference copies are committed under `expected/` and `figures/`.

## Check that it reproduces

```
python verify.py
```

This copies the model to a scratch directory, runs it, and compares both the stdout log and the CSV against the reference artifacts. It exits non-zero on any difference, and leaves the working tree clean.

The run is deterministic. A fixed seed feeds per-replication seeds derived arithmetically rather than through `hash()`, which Python randomizes per process and which would otherwise have made runs vary.

Figures are not byte-compared, because matplotlib output shifts between versions and a pixel difference is not a reproduction failure. The CSV carries the data.

## Scope and limits

Worth reading before citing any number from this repository.

- **It is not a measurement.** All three delivery architectures are constructed baselines parameterized from a table, not deployments. The output compares them under a stated model. It does not predict absolute completion rates for any real fleet.
- **The atomicity assumption drives the result.** A fully resumable download regime reduces completion to an aggregate bandwidth condition. The effect studied here is specific to the atomic install regime, which is why resumability is swept rather than assumed away.
- **Most window parameters are assumptions.** Only the drone flight duration is literature-grounded. Dock and parking windows and the local-to-backhaul rate ratio are representative values, and their influence is bounded by the sweeps rather than eliminated.
- **Gossip is a logistic epidemic model** over a sparse, independently tasked fleet. The contact-rate sweep shows where it would become competitive.

## Version history

The model was revised twice for correctness, both times after review found the earlier version was measuring the wrong thing. `CHANGELOG.md` records what changed and why. `archive/` holds the superseded version with its artifacts.

## Citation

See `CITATION.cff`. Update it on acceptance.

## License

MIT, see `LICENSE`.
