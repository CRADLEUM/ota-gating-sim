# Model (v4)

Reference for what `gating_sim.py` computes. Read alongside the source, which is one file and stays close to this description. The v3 description is in `archive/` with the v3 source.

## State machine

Each of N devices alternates between two states:

- **Operation.** The device is in public space. It cannot install.
- **Stationary.** The device is docked or parked. It can receive and install.

Window durations come from one of three sources, chosen per run:

| Source | Operation window | Stationary window | Randomness |
| --- | --- | --- | --- |
| `exp` | exponential, v3 mean | exponential, v3 mean | yes |
| `bootstrap` | resampled from measured flight or trip durations | resampled from measured dock or curb gaps (daytime start) | yes |
| `replay` | the device's real schedule | the device's real schedule | none |

Measured sets are in `data/sim_inputs/` and come from `scripts/make_sim_inputs.py`: drone windows per agency from DFR flight logs under the minimum fleet consistent with the overlaps (a lower bound on dock windows), AV windows from Chicago taxi trips. Both floors and sources are documented in the paper's evidence file.

At t = 0 a device is in operation with probability `mean_op / (mean_op + mean_stat)`, the stationary distribution of the alternating renewal process. Its remaining time in that state is the residual of the window in progress. For exponential windows the residual has the same distribution as the window, and the elapsed part does too, so the full length of the window in progress is their sum. For measured windows the window in progress is drawn length-biased (probability proportional to length) and the position within it is uniform, which is the renewal-theory residual. The depot rule below is applied to the full length of that window, not to its remaining part, so a long stop already under way at release counts as a depot stop. In replay there is no draw: the campaign starts at a stated clock time and each device is wherever its log puts it.

## Completion

A device becomes current only when a **single** stationary window is long enough to do both of the following, contiguously:

1. Transfer the whole update at the in-window delivery rate.
2. Complete an atomic install lasting `install_t` minutes.

Leaving a stationary window discards the partial session. Transfer progress carries across the boundary only in proportion to a resumable fraction ρ, which defaults to 0. Install progress never carries.

Units: update size S in GB, link rates in Mbps, time in minutes. Transfer time is `S × 8000 / rate / 60`.

The reported metric is the **completion-before-redeployment rate**: the fraction of devices current by a deadline H after release. v4 reports it along two axes:

- against S at the six-hour deadline, summarised by **S95**, the largest swept S at which the architecture still reaches `TARGET` (0.95), with no margin;
- against the deadline at anchor sizes, summarised by **T95**, the earliest deadline on a 10-minute grid at which completion reaches `TARGET`.

At measured 2026 link rates a multi-gigabyte transfer takes minutes, so S95 sits at or beyond the top of the sweep for most settings. T95 is where the install gate, the window distribution, and reach show up. The v3 margin crossover (first S where depot leads the best gating-unaware architecture by more than 0.15) is printed once for continuity.

## Reach

A stationary window is a **depot window** when the device sits where local staging exists. Rule per class:

| Rule | Meaning | Default |
| --- | --- | --- |
| `("all", None)` | every stationary window is at the dock | drones |
| `("long", m)` | windows of at least m minutes are depot or garage stops | AVs, m = 120 (the data: 19 percent of daytime gaps, 80 percent of gap time) |
| `("prob", p)` | each window is a depot window with probability p | reach sweep |

This is the reach-versus-rate trade in the paper: cellular or satellite push reaches every window, local staging has the rate but only where it exists.

## Delivery architectures and baselines

| Name | In-window behaviour |
| --- | --- |
| `centralized` | transfers at the backhaul rate `r_bh` in every stationary window |
| `depot` | transfers at the local rate `r_local` in depot windows only |
| `hybrid` | uses the better link available in each window (the faster of `r_local` and `r_bh` at a depot, `r_bh` elsewhere) |
| `gossip` | holds the full update the moment the device is infected; the install gate still applies |
| `nonatomic` | baseline selector: a scheduler that knows the windows but lets transfer span them (ρ = 1 for transfer); install still atomic |
| `bandwidth` | baseline selector: completion if backhaul rate × total stationary time covers transfer plus install, windows ignored |

Gossip infection times come from the logistic SI curve `I(t) = I0 / (I0 + (1 − I0) e^{−βt})`; each device inverts it at a uniform draw (bootstrap) or at the quantile `(i + 0.5) / N` (replay, deterministic). β is contacts per node per minute with transmission on every contact, an upper bound for a fleet of the measured density.

The two baselines answer the review: the report lists the deadlines (and sizes) at which each baseline's model says centralized push meets the target while atomic completion does not.

## Peer links and the contact gate

The gossip rate β counts every approach within range as a transfer, which is an upper bound. A whole-image transfer over a peer link needs a contact at least as long as the transfer at that link's rate, so the run also reports a **contact gate**: for each documented peer-link rate r and size S, the transfer time `S × 8000 / r / 60`, the share of measured Cabspotting contacts at 100 m (daytime) at least that long, and the gated rate `β_eff = β × share`. Gossip T95 is then run at `β_eff` for the firmware anchor (0.42 GB) and the headline (4 GB). The contact histogram has 30-second bins with an overflow bin at one hour, and a bin counts only if it starts at or after the threshold, so the share is conservative by less than one bin; transfers longer than an hour get a share of zero.

The rates are documented, not assumed (`data/derived/documented_rates.json`, `peer_links`):

| Class | Link | Rate (Mbps) | Kind |
| --- | --- | --- | --- |
| AV | LTE-V2X PC5, field, four nodes on 20 MHz, commercial radios | 0.72 mean, about 2.6 peak | measured |
| AV | IEEE 802.11p, 10 MHz channel | 6 default, 27 top of the rate set | spec |
| AV | 5G NR sidelink | not used | no field figure found |
| drone | MANET mesh radios flown on UAVs (Silvus SC4200E+, Doodle Labs) | up to 100, up to 80 | vendor spec |
| drone | four-UAV SDR mesh demonstration | 99.84 per link | research demo |

Drones have no contact data (no trace exists for docked fleets, and the deployed dock systems have no aircraft-to-aircraft link), so the drone rows give only the contact length a transfer would need.

The run also prints a **transfer-time table** for both classes: at 0.42 and 4 GB, the minutes each documented or measured link needs (cellular backhaul, satellite, poor cellular, local staging, dock Ethernet, and the peer links), the share of measured daytime stationary windows at least as long as transfer plus install (one window's chance of completing), and for the peer links the share of measured contacts at least as long as the transfer.

## Parameters

Fleet-wide:

| Symbol | Value | Meaning |
| --- | --- | --- |
| `SEED` | 20260705 | base seed |
| `H_MAX` | 360 min | longest deadline, the v3 horizon |
| `H_GRID` | 10 to 360 step 10 | deadlines for the T95 curves |
| `N` | 500 | devices per class |
| `R`, `R_SENS` | 60, 30 | replications, main and sensitivity |
| `TARGET` | 0.95 | completion target behind S95 and T95 |
| `S_GB` | 0.25 to 10 step 0.25 | update sizes at H_MAX |
| `S_ANCHORS` | 0.5, 2, 4, 8 GB | sizes for the deadline curves; 4 GB is the headline (documented dock update order of 3.75 GB) |

Per class (defaults; every one is swept):

| Parameter | drone | av | Basis |
| --- | --- | --- | --- |
| window set | Cincinnati PD docks, K_min fleet, daytime | Chicago taxi, daytime | measured |
| `install_t` | 20 min | 30 min | documented (DroneDeploy; Tesla manual) |
| `r_bh` | 137 Mbps (rural cellular) | 297 Mbps (Chicago metro cellular) | Ookla 2026 Q2 |
| `r_local` | 372 Mbps (fixed broadband) | 372 Mbps | Ookla 2026 Q2 |
| depot rule | all | long, 120 min | data |
| `gossip_beta` | 0.0015 per min (v3 assumption; no contact data exists for docked fleets) | 0.822 per min (Cabspotting, 536 taxis, 100 m range, daytime, transmission on every contact) | measured for AVs, `data/derived/cabspotting`; the sweep gates it by contact length |
| `gossip_seed` | 0.05 | 0.05 | assumed |
| exp means (v3) | 30 / 60 min | 180 / 120 min | v3 assumptions, kept for the comparison run |

## Sweeps

Each varies one quantity on bootstrap windows with everything else at the class default, and reports T95 at the headline size.

| Sweep | Varies | Purpose |
| --- | --- | --- |
| install time | 5 to 60 min | the documented install is the binding gate |
| backhaul | 5 to 297 Mbps, anchors marked | where the delivery path starts to matter |
| gate isolation | stationary windows scaled 0.25 to 2, rates fixed | window length alone moves the answer |
| reach (AV) | p from 0 to 1 | local staging without reach |
| gossip | β from the v3 assumptions (0.0015, 0.006) through the duration-gated measured rates (contacts of at least 20, 10, 5, 2, 1 min: 0.013 to 0.523) to every contact (0.822) | where gossip becomes competitive, and what contact length it needs |
| peer link (AV) | documented vehicle-to-vehicle rates, at 0.42 and 4 GB, through the contact gate | whether any documented peer link carries the image inside measured contacts |
| agencies | five DFR agencies | population sensitivity |
| resumability | ρ from 0 to 1, exact window-walking engine | how much survives resumable transfer |
| replay start hour | 08, 12, 16, 20 | diurnal effect, deterministic |

## Engines and checks

For ρ = 0 completion is evaluated exactly on each device's window sequence: a device completes by deadline H if some stationary window, clipped to [0, H], is at least `transfer + install` long (with the architecture's rule for which windows and which rate). No time stepping.

For ρ > 0 a second exact engine walks the windows in order: transfer progress carries across a window boundary as ρ times the progress at the end of the window, a device that already holds the whole image installs at any stop that is at least `install_t` long (the install needs no link), and install progress never carries. At ρ = 0 this reproduces the main engine for all three architectures; at ρ = 1 with centralized push it reproduces the window-aware non-atomic baseline. The run prints both differences, which are zero to floating point because both engines see the same window sequences.

The v3 one-minute stepped engine is kept as an independent cross-check on its own random stream. It credits whole steps, so a window earns up to one extra minute and the step in which the transfer completes also counts towards the install; it runs about two minutes optimistic per window. The run prints its largest deviation from the exact engine at ρ = 0, which is statistical (different seeds) plus that bias.

The renewal closed form `1 − (1 − e^{−τ/m_s})^{H/(m_o + m_s)}` from v3 is compared against the exponential-window run. It fixes the number of windows at `H / (m_o + m_s)` and ignores both the window under way at release and the deadline cut, so it is a long-horizon approximation: wrong by up to 0.2 inside the first cycle, within about 0.01 at six hours. The run prints the deviation at six hours, beyond two cycles, and overall.

## Data cleaning

Windows are positive intervals. `make_sim_inputs.py` drops zero-length records before resampling (764 of about 972,000 taxi windows: gaps of exactly zero between adjoining trips, and trips with no duration; none in the drone logs) and records the count in `data/sim_inputs/README.json`. Chicago taxi trip starts are rounded to 15 minutes in the public data, so about 7 percent of consecutive trips of one taxi overlap on paper; in replay the engine clips such gaps to zero, which treats the pair as one stretch of operation. Short windows that are positive (a 2-second flight log entry, a 1-minute gap) stay in, as measured.

## Tests

`python -m pytest tests/` (about 5 s). The suite checks the unit conversion, the sequence geometry and the length-biased window in progress, every architecture and baseline against a plain per-device reference implementation on random sequences, the ordering invariants (bandwidth-only ≥ non-atomic ≥ centralized; hybrid ≥ centralized and ≥ depot), monotonicity in size, deadline and ρ, the two exact engines against each other and against the reference, the stepped engine within its stated bias, a two-device synthetic replay with hand-computed completion, the S95 and T95 summaries, the SI inversion, and the determinism of the seeds.

## Determinism

Every replication gets its own seed, derived arithmetically:

```
SEED + r * 1_000_003 + class_index * 10_007 + source_index * 101 + tag * 7_919
```

Nothing calls `hash()`. Runs reproduce across machines with a compatible numpy random stream, which `verify.py` checks. The replay runs contain no random draws at all.

## Output schema

`results_v4.csv`, one row per class, window source, architecture, update size, and deadline:

| Column | Meaning |
| --- | --- |
| `class` | `drone` or `av` |
| `source` | `exp`, `bootstrap`, or `replay12` (campaign start 12:00) |
| `arch` | architecture or baseline name |
| `S_gb`, `H_min` | update size and deadline; rows at `H_min = 360` span `S_GB`, rows at anchor sizes span `H_GRID` |
| `mean`, `std` | across replications (std is 0 for replay) |

`sweeps_v4.csv` holds every sweep's T95 (minutes) and completion at 360 min. `replay_v4.csv` holds the replay curves for every start hour, over all device-days and over active device-days only.
