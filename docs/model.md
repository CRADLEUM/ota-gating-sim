# Model

Reference for what `gating_sim.py` computes. Read alongside the source, which is one file and stays close to this description.

## State machine

Each of N devices alternates between two states:

- **Operation.** The device is in public space. It cannot install. Mean duration `op_mean`.
- **Stationary.** The device is docked or parked. It can receive and install. Mean duration `stat_mean`.

Both durations are exponential, floored at one minute. Time advances in one-minute steps to a horizon H.

At t=0 a device is in operation with probability `op_mean / (op_mean + stat_mean)`, which is the stationary distribution of the alternating renewal process. Its remaining time in that state is drawn from the same exponential as the state itself, since the residual life of an exponential window is exponential with the same mean. Getting this wrong was the v3 fix.

## Completion

A device becomes current only when a **single** stationary window is long enough to do both of the following, contiguously:

1. Transfer the whole update at the in-window delivery rate.
2. Complete an atomic install lasting `INSTALL_T` minutes.

Leaving a stationary window discards the partial session. Transfer progress carries across the boundary only in proportion to a resumable fraction ρ, which defaults to 0. Install progress never carries.

The reported metric is the fraction of devices current by H, called the completion-before-redeployment rate.

Update size S is measured in delivery-minutes, so a rate is delivery-minutes per minute and `S / rate` is a transfer time in minutes.

## Delivery architectures

| Topology | In-window behaviour |
| --- | --- |
| `centralized` | Transfers at `cen_r_stat`, the wide-area backhaul rate at the dock or curb |
| `depot` | Transfers at `depot_r`, a local staging rate independent of backhaul |
| `gossip` | Holds the full update the moment the device is infected, so transfer is not the constraint; the install gate still applies |

Gossip infection times come from a logistic epidemic. With initial infected fraction `I0` and contact rate β, the infected fraction at time t is `I0 / (I0 + (1 - I0) e^{-βt})`. Each device draws a uniform and inverts that curve to get its infection time.

## Parameters

Fleet-wide:

| Symbol | Value | Meaning |
| --- | --- | --- |
| `SEED` | 20260705 | Base seed |
| `H` | 360 min | Patch deadline, six hours |
| `DT` | 1 min | Time step |
| `N` | 500 | Devices per class |
| `INSTALL_T` | 5 min | Atomic install duration |
| `R` | 60 | Replications per configuration, main sweep |
| `R_SENS` | 30 | Replications per configuration, sensitivity sweeps |
| S values | 5 to 195 step 10 | Update sizes, delivery-minutes |

Per class:

| Parameter | drone | av |
| --- | --- | --- |
| `op_mean` | 30 min | 180 min |
| `stat_mean` | 60 min | 120 min |
| `cen_r_stat` | 0.30 | 1.10 |
| `depot_r` | 4.0 | 4.0 |
| `gossip_beta` | 0.0015 /min | 0.006 /min |
| `gossip_seed` | 0.05 | 0.05 |

The drone flight duration is the one literature-grounded value. The rest are representative assumptions, and the sweeps below bound their influence rather than removing it.

## Crossover

`S*` is the smallest swept update size at which depot-staged completion exceeds the better of the two gating-unaware architectures by more than 0.15. Reported as `None` when no such size exists in the swept range, which the resumability panel plots at a capped value with a label rather than dropping.

The 0.15 threshold is a reporting choice, not a property of the model. Reading `results_v3.csv` directly avoids it.

## Sweeps

Four, each varying one quantity with everything else held fixed.

| Sweep | Varies | Holds fixed | Purpose |
| --- | --- | --- | --- |
| Gate isolation | Drone `stat_mean`, 20 to 320 min | All transfer rates | Tests whether window length alone moves the preferred architecture |
| Backhaul | Drone `cen_r_stat`, 0.10 to 1.60 | Window length | Sensitivity to the link rate |
| Resumability | ρ, 0 to 1 | Everything else | Tests how much of the effect survives resumable transfer |
| Gossip contact rate | β, 0.0015 to 0.030 /min | S = 105 | Locates where gossip would become competitive |

Gate isolation is the one that carries the argument. If holding every rate constant and moving only the window changes the answer, then window length is an independent driver.

## Closed-form check

For the atomic model, a device completes if some stationary window within H is at least `τ = S / r + INSTALL_T` long. With exponential windows, `P(window ≥ τ) = e^{-τ / stat_mean}`, and the expected number of stationary windows in H is `H / (op_mean + stat_mean)`. That gives

```
completion ≈ 1 - (1 - e^{-τ / stat_mean}) ^ (H / (op_mean + stat_mean))
```

The script prints the largest absolute deviation between this approximation and the simulated means. It is a check on the simulator, not an independent result. The approximation treats window count as deterministic and ignores the partial window in progress at the horizon, so exact agreement is not expected.

## Determinism

Every replication gets its own seed, derived arithmetically:

```
SEED + r * 1_000_003 + class_index * 10_007 + topology_index * 101 + int(S) + tag * 7_919
```

The `tag` separates sweeps that would otherwise reuse the same coordinates. Nothing calls `hash()`, which Python randomizes per process and which would make runs vary between invocations.

Two consequences. Runs are reproducible across machines with a compatible numpy random stream, which `verify.py` checks. And configurations that share coordinates across sweeps share a random stream, which is common variance reduction for a sweep but means neighbouring points are not statistically independent.

## Output schema

`results_v3.csv`, one row per class and update size:

| Column | Meaning |
| --- | --- |
| `class` | `drone` or `av` |
| `S` | Update size, delivery-minutes |
| `cen_mean`, `cen_std` | Centralized push, mean and standard deviation over R replications |
| `gossip_mean`, `gossip_std` | P2P gossip |
| `depot_mean`, `depot_std` | Depot-staged |

Standard deviations are across replications, so a 95 percent band is roughly `mean ± 1.96 × std`. The sweeps print to stdout and are not written to CSV.
