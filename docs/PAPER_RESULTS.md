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
6. [PPO trainability sweep across NE-class cells (paper §4)](#6-ppo-trainability-sweep-across-ne-class-cells-paper-4)

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

## 6. PPO trainability sweep across NE-class cells (paper §4)

**Result**: Per-cell PPO trainability scan over the same $(\beta, \kappa, c)$
grid as §4. The headline finding ordered by NE class is
$\textsf{symmetric\_only}~(+0.180; n{=}11)$ $>$
$\textsf{mixed}~(+0.107; n{=}9)$ $>$
$\textsf{asymmetric\_only}~(+0.059; n{=}11)$ $>$
$\textsf{no\_convergence}~(-0.049; n{=}6)$ on the
`gap_closed_ne` metric (Random$\to$1$\times$Hero$+$3$\times$Firefighter
recalibrated baseline). A targeted 4$\times$-budget rerun on the 6
`no_convergence` cells worsens the verdict from $-0.049$ to $-0.108$,
ruling out under-training as the cause of PPO's collapse on those cells.
This is the trainability evidence the workshop paper's §4 finding rests on.

**Artifacts**:
- Original-budget sweep: `experiments/p3_specialization/phase_diagram_ppo_v2/cell_<tag>/cell_summary.json` (37 cells × 4 seeds)
- 4×-budget rerun: `experiments/p3_specialization/phase_diagram_ppo_longbudget/cell_<tag>/cell_summary.json` (6 no_convergence cells × 4 seeds)
- Recalibrated verdict tables: [`experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.md`](../experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.md), [`experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.md`](../experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.md)
- Per-cell baselines consumed by the recalibrator: [`experiments/nash/phase_diagram/per_cell_baselines.json`](../experiments/nash/phase_diagram/per_cell_baselines.json)
- Figure 2 source (per-cell heatmap): [`paper/anvil_pub.bb-workshop.4/figures/src/recalibrated_heatmap.py`](../paper/anvil_pub.bb-workshop.4/figures/src/recalibrated_heatmap.py)
- Published paper: [`paper/anvil_pub.bb-workshop.4/main.tex`](../paper/anvil_pub.bb-workshop.4/main.tex)

### 6a. PPO sweep at original budget (37 cells × 4 seeds)

**Reproduce** **`[remote]`** — ~6h on a 32-core host (the canonical run was
on alc-2). The launcher resolves a target host from `.env`, SSHes in,
pulls latest main, rebuilds the Rust extension, and starts a detached
tmux session that drives `run_phase_diagram_ppo.py` cell by cell. Per-cell
summaries land at `cell_<tag>/cell_summary.json`:

```bash
./experiments/scripts/launch_phase_diagram_ppo.sh
```

### 6b. PPO sweep at 4× budget on no_convergence cells (6 cells × 4 seeds)

**Reproduce** **`[remote]`** — ~6h on a 32-core host (the canonical run was
on alc-6). There is no dedicated launcher yet; invoke the driver directly
inside a remote tmux session, after constructing a 6-cell subset of
`experiments/nash/phase_diagram/results.json` containing only cells with
`verdict == "no_convergence"`:

```bash
python3 experiments/p3_specialization/run_phase_diagram_ppo.py \
    --cells-source /tmp/no_conv_only.json \
    --scenario minimal_specialization \
    --seeds 42 43 44 45 \
    --num-iterations 200 --rollout-steps 4096 \
    --output-root experiments/p3_specialization/phase_diagram_ppo_longbudget
```

### 6c. Recalibration (no compute)

The raw `cell_summary.json` files use a single canonical
MINSPEC_RANDOM / MINSPEC_SPECIALIST pair across all cells. The
recalibrator re-aggregates against per-cell Random and per-cell
1$\times$Hero$+$3$\times$Firefighter baselines, producing the
`gap_closed_homogeneous` and `gap_closed_ne` columns the paper reports.
Run once per sweep:

```bash
uv run python experiments/scripts/recalibrate_phase_diagram_ppo.py \
    --ppo-root experiments/p3_specialization/phase_diagram_ppo_v2 \
    --output-md experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.md \
    --output-json experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json

uv run python experiments/scripts/recalibrate_phase_diagram_ppo.py \
    --ppo-root experiments/p3_specialization/phase_diagram_ppo_longbudget \
    --output-md experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.md \
    --output-json experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.json
```

### 6d. Headline-table and Figure 2 generation (no compute)

The §4 heatmap (Figure 2) reads `recalibrated_verdict.json` from the
original-budget sweep and renders a 3-panel 3×5 grid (β × κ across
$c \in \{0.5, 1.0, 2.0\}$) per-cell `gap_closed_ne` heatmap to vector PDF:

```bash
uv run --with matplotlib python \
    paper/anvil_pub.bb-workshop.4/figures/src/recalibrated_heatmap.py
```

The 4-class ordering summary printed at the bottom of each
`recalibrated_verdict.md` is the textual headline that grounds the paper's
§4 prose.

### 6e. Workshop paper cross-link

For the surrounding argument — the closed-form bound that predicts
collapse $\to$ symmetric $\to$ asymmetric, how the empirical
`mixed` class sits between symmetric and asymmetric, and the
trainability-vs-NE-structure framing — see
[`paper/anvil_pub.bb-workshop.4/main.tex`](../paper/anvil_pub.bb-workshop.4/main.tex)
§4.

### 6f. k = 1 improvability oracle on the `no_convergence` cells (issue #428)

**Result**: A downstream consumer (issue #428, via the `rjwalters/thrust`
harness) reported that the $\kappa = 0.1$ / $c = 0.5$ `no_convergence`
cells are *degenerate for single-best-response learning* — i.e. that a
single agent facing 3 frozen uniform-random opponents has essentially no
improvable gap ($\leq +0.01\%$ team-return headroom, achieved by resting).
The repo-native k = 1 oracle does **not** reproduce this on the repo's
`minimal_specialization`-based phase-diagram cells: a deterministic
any-house firefighter beats the all-uniform baseline by **+15.3%** on
per-step team return (paired 95% CI on the delta $[+11.1, +17.7]$ per
step, n = 400 paired episodes), and the hand-coded specialist is
significantly *better* than uniform ($+4.7$ per step), not worse. The
scale mismatch in the report ($\approx 7\times$ larger return magnitudes)
indicates the downstream harness evaluated a differently-weighted reward
configuration; see the non-reproduction note in the committed artifact.

**Interpretation for §4/§6**: flat PPO returns on the three
$c = 0.5$ `no_convergence` cells cannot be read as "no single-agent gap
exists, so flat is correct" — a scripted k = 1 best response finds a
statistically decisive team-return improvement that PPO does not. The
trainability failure on these cells therefore remains unexplained by this
oracle (as it is by NE conditional entropy, §430 Task 1); the
coordination-threshold account ($k^* > 1$, issue #430 / thrust#259) and
plain exploration failure remain the live hypotheses. The three
$c = 2.0$ `no_convergence` cells are not yet characterized (run the
oracle with `--cells b0.10_k0.10_c2.00 b0.50_k0.10_c2.00
b0.90_k0.10_c2.00` in a follow-up).

**Artifacts**:
- Oracle script: [`experiments/nash/phase_diagram/improvability_oracle.py`](../experiments/nash/phase_diagram/improvability_oracle.py)
- Committed results: [`experiments/nash/phase_diagram/improvability_oracle.json`](../experiments/nash/phase_diagram/improvability_oracle.json), [`experiments/nash/phase_diagram/improvability_oracle.md`](../experiments/nash/phase_diagram/improvability_oracle.md)

**Reproduce** (scripted policies only — no training; safe locally, ~20 s
on a multi-core machine):

```bash
uv run python experiments/nash/phase_diagram/improvability_oracle.py --random-search 64
```

**Source**: PR #420 (sweep results + per-cell baselines + recalibrator),
PR #421 (workshop paper that uses this data), parent #357 (M4 release-
infra tracker), issue #428 (k = 1 oracle, §6f).

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
