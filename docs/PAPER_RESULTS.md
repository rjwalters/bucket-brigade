# Bucket Brigade — Paper Results

Reproduction guide for the figures and tables that ship with the
Bucket Brigade workshop paper. Each section lists the in-repo artifact
paths, the one-line command to regenerate the result, and any compute
caveats. All commands assume:

- The repo is cloned and `uv sync --extra rl` has been run.
- The Rust extension is built (`bash bucket-brigade-core/build.sh`) —
  Nash and evolution runs need the ~100× speedup or the wall-clock
  budgets explode.
- Commands marked **`[remote]`** are designed for a multi-core remote
  host. See [`experiments/REMOTE_EXECUTION.md`](../experiments/REMOTE_EXECUTION.md)
  and [`CLAUDE.md`](../CLAUDE.md) for the cluster workflow.

## Index

1. [Environment specification (paper §2)](#1-environment-specification-paper-2)
2. [Benchmark comparison (paper §1)](#2-benchmark-comparison-paper-1)
3. [Heterogeneous Nash equilibria (paper §3)](#3-heterogeneous-nash-equilibria)
4. [Nash phase diagram across (β, κ, c)](#4-nash-phase-diagram-across-β-κ-c)
5. [Specialist exploitability harness](#5-specialist-exploitability-harness)

---

## 1. Environment specification (paper §2)

**Result**: The formal mathematical contract for the Bucket Brigade
environment — player set, parameter family, state/action/observation
spaces with exact cardinalities, seven-phase night dynamics, reward
structure, and the relationship to canonical game-theory templates
(Volunteer's Dilemma, N-player Public Goods, Stag Hunt, stochastic
games). Self-contained: a reader unfamiliar with the codebase can
reimplement the env from this document alone.

**Artifact**: [`paper/anvil_memo.env_spec.1/env_spec.md`](../paper/anvil_memo.env_spec.1/env_spec.md)

**How to verify**: The cardinalities cited in the spec
(2304 states and 4096 joint actions at `H=2, N=4`) can be checked
directly:

```bash
uv run python -c "
import bucket_brigade
env = bucket_brigade.make('v2_minimal-v1')
print('obs space:', env.observation_space)
print('action space:', env.action_space)
print('joint action cardinality:', int(env.action_space.nvec.prod()))
"
```

For the Python-side env API reference, see [`docs/ENV.md`](ENV.md).

**Source**: PR #375 / issue #362.

---

## 2. Benchmark comparison (paper §1)

**Result**: Positioning of Bucket Brigade against the canonical
cooperative-MARL benchmarks (Overcooked, Hanabi, SMAC, Melting Pot,
MAgent, PettingZoo MPE). The two distinguishing properties are
(a) state and joint-action spaces small enough to enumerate at minimal
parameterizations, so Nash equilibria can be computed exactly rather
than approximated, and (b) load-bearing parameters $(\beta, \kappa, c)$
that move equilibrium structure continuously between cooperative-dominant
and free-rider-dominant regimes.

**Artifact**: [`paper/anvil_report.benchmark_comparison.1/`](../paper/anvil_report.benchmark_comparison.1/)

**How to verify**: Read the report — no computation to rerun.

---

## 3. Heterogeneous Nash equilibria

**Result**: Heterogeneous Double-Oracle Nash analysis on two
canonical scenarios resolves the symmetric-vs-asymmetric question
that symmetric DO was failing on.

| Scenario | Verdict | Best converged NE | Pattern |
|---|---|---|---|
| `minimal_specialization-v1` | `symmetric_ne_superior` | **−756** (sym Hero) | 4/5 converged restarts are pure all-Hero |
| `rest_trap-v1` | `asymmetric_only` | **+2984** (asym FR/FF) | 8/13 converged restarts are `FR × 3 + FF × 1` |

Role-differentiated specialization is **not** a Nash equilibrium on
`minimal_specialization`; the symmetric all-Hero strategy dominates any
asymmetric deviation. `rest_trap` admits **zero** symmetric NE — the
free-rider equilibrium (three free-riders plus one firefighter) is the
only stable profile.

**Artifacts**:
- Headline writeup: [`experiments/nash/heterogeneous/RESULTS.md`](../experiments/nash/heterogeneous/RESULTS.md)
- Per-scenario summaries: [`experiments/nash/heterogeneous/minimal_specialization/summary.md`](../experiments/nash/heterogeneous/minimal_specialization/summary.md), [`experiments/nash/heterogeneous/rest_trap/summary.md`](../experiments/nash/heterogeneous/rest_trap/summary.md)
- Raw equilibrium results: `experiments/nash/heterogeneous/{minimal_specialization,rest_trap}/results.json`

**Reproduce** **`[remote]`** — full sweep is ~5–7h per scenario on a
32-thread CPU box:

```bash
uv run python experiments/scripts/compute_nash_heterogeneous.py minimal_specialization \
    --restarts 20 --simulations 1000 --opt-simulations 300 \
    --max-iterations 25 --epsilon 50 --seed 42

uv run python experiments/scripts/compute_nash_heterogeneous.py rest_trap \
    --restarts 20 --simulations 1000 --opt-simulations 300 \
    --max-iterations 25 --epsilon 50 --seed 42
```

**Regenerate verdict tables from existing `results.json`** (no compute):

```bash
uv run python experiments/nash/heterogeneous/regen_summaries.py
```

**Source**: PRs #354 / #355 (sweep + verdict-logic fix).

---

## 4. Nash phase diagram across (β, κ, c)

**Result**: Verdict map of heterogeneous Nash structure across the
$(\beta, \kappa, c)$ grid. The $c=0.5$ plane shows a clean κ-driven
phase pattern: low κ collapses to `no_convergence` (negative equilibrium
payoff, no one extinguishes), mid κ admits `symmetric_only` NE, and high
κ pushes the system into `asymmetric_only` free-rider territory.
β does not move the regime within the partial slice that has been
filled — the phase boundary at $c=0.5$ is one-dimensional in κ.

Partial preview (7 of 18 design cells at $c=0.5$):

| β\κ  | 0.10 | 0.50 | 0.90 |
|------|------|------|------|
| 0.10 | `no_convergence` | (missing) | (missing) |
| 0.50 | `no_convergence` | `symmetric_only` | `asymmetric_only` |
| 0.90 | `no_convergence` | `symmetric_only` | `asymmetric_only` |

The frequency-cap validation rerun (alc-5 at 4.5 GHz) returned
bit-identical verdicts, payoffs, and convergence rates to the
unconstrained alc-2 host — confirming the solver is deterministic given
the seed and that the cap is a correctness-preserving stability
workaround for the Raptor Lake degradation crashes observed during the
preview run.

**Artifacts**:
- Figures: `experiments/nash/phase_diagram/phase_diagram.png`, `experiments/nash/phase_diagram/phase_diagram_freqtest.png`
- Tables: [`experiments/nash/phase_diagram/phase_diagram_table.md`](../experiments/nash/phase_diagram/phase_diagram_table.md), `experiments/nash/phase_diagram/phase_diagram_table_freqtest.md`
- Recovery writeup: [`experiments/nash/phase_diagram/RECOVERY_NOTES.md`](../experiments/nash/phase_diagram/RECOVERY_NOTES.md)
- Per-cell raw outputs: `experiments/nash/phase_diagram/preview/<host>/cells/*/summary.json`
- Aggregate inputs: `experiments/nash/phase_diagram/results.json`, `results_freqtest.json`

**Reproduce a single cell** **`[remote]`** — ~5.5h per cell at design
budgets, ~25–30 min for a non-converging restart:

```bash
uv run python experiments/scripts/compute_nash_phase_diagram.py \
    --beta-values 0.5 --kappa-values 0.9 --c-values 0.5 \
    --restarts 20 --simulations 1000 --opt-simulations 300 \
    --max-iterations 25 --epsilon 50 --seed 42 \
    --num-workers $(nproc)
```

**Regenerate figures and tables from `results.json`** (no compute):

```bash
uv run python experiments/scripts/plot_phase_diagram.py \
    experiments/nash/phase_diagram/results.json
```

**Fill gaps using the runbook script** **`[remote, multi-host]`** —
see [`experiments/nash/phase_diagram/LAUNCH_RUNBOOK.md`](../experiments/nash/phase_diagram/LAUNCH_RUNBOOK.md)
for the three-plan operator workflow:

```bash
./experiments/scripts/launch_phase_diagram_fill.sh \
    --beta-values 0.1 --kappa-values 0.5,0.9 --c-values 0.5 \
    --num-workers 32
```

**Source**: PRs #387 (preview results + recovery notes), #391
(per-restart checkpointing), #392 (parallel Pool fix that gets cluster
hosts saturating), #393 (launch runbook + script).

---

## 5. Specialist exploitability harness

**Result**: Per-position best-response check that verifies a 4-player
heterogeneous Nash profile is exploitation-robust. For each position,
the harness holds the other three positions fixed at the equilibrium
strategy, runs a 5-archetype grid scan plus L-BFGS-B refinement on the
10-d parameter vector, and flags a position as exploitable iff
`BR_payoff − equilibrium_payoff > ε`.

The harness is shipped as a *script* (commit-merged); the full
verdict-producing sweep is a remote-only multi-hour job that operators
launch manually per scenario.

**Artifact**: [`experiments/scripts/test_specialist_exploitability.py`](../experiments/scripts/test_specialist_exploitability.py)

**Reproduce** **`[remote]`** — ~1.5h per scenario on a 32-thread box,
writes `experiments/nash/heterogeneous/<scenario>/exploitability.json`
with per-position `{equilibrium_payoff, best_response_payoff, gap,
exploitable_flag, best_response_closest_archetype}` plus an aggregate
`any_exploitable` flag (and non-zero exit if any flag is true):

```bash
uv run python experiments/scripts/test_specialist_exploitability.py rest_trap
uv run python experiments/scripts/test_specialist_exploitability.py minimal_specialization
```

**Validate the harness wires correctly without paying the full cost**:

```bash
uv run python experiments/scripts/test_specialist_exploitability.py rest_trap \
    --smoke --simulations 20 --opt-simulations 10 --positions 0
```

A companion symmetric-NE verification harness lives at
[`experiments/scripts/test_specialist_nash.py`](../experiments/scripts/test_specialist_nash.py)
(PR #378). It does a best-response chain in heuristic-parameter space
against the `minimal_specialization` specialist baseline.

**Source**: PRs #376 (per-position exploitability harness), #378
(symmetric specialist Nash chain).

---

## Repro environment

A clean reproduction sequence on a fresh machine:

```bash
git clone https://github.com/rjwalters/bucket-brigade.git
cd bucket-brigade
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra rl
bash bucket-brigade-core/build.sh   # ~100x speedup for Nash/evolution
uv run pytest tests/                 # sanity check the install
```

For the Python env API, see [`docs/ENV.md`](ENV.md). For training, see
[`docs/TRAINING_GUIDE.md`](TRAINING_GUIDE.md). For Nash benchmarking
provenance, see [`docs/NASH_BENCHMARKS.md`](NASH_BENCHMARKS.md). For the
formal env spec referenced from §2 of the paper, see
[`paper/anvil_memo.env_spec.1/env_spec.md`](../paper/anvil_memo.env_spec.1/env_spec.md).
