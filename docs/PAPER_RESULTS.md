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
7. [Trap-escape verdict rule for social-trap scenarios (rest_trap first)](#7-trap-escape-verdict-rule-for-social-trap-scenarios-rest_trap-first)
8. [het_ppo Phase 2 on the asymmetric_only cells: at_random](#8-het_ppo-phase-2-on-the-asymmetric_only-cells-at_random)
9. [Trap-escape ladder rung 1: budget scaling on rest_trap (4×/16×)](#9-trap-escape-ladder-rung-1-budget-scaling-on-rest_trap-416)

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
- Figure 2 source (per-cell heatmap): [`paper/anvil_pub.bb-workshop.8/figures/src/recalibrated_heatmap.py`](../paper/anvil_pub.bb-workshop.8/figures/src/recalibrated_heatmap.py)
- Published paper: [`paper/anvil_pub.bb-workshop.8/main.tex`](../paper/anvil_pub.bb-workshop.8/main.tex)

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
    paper/anvil_pub.bb-workshop.8/figures/src/recalibrated_heatmap.py
```

The 4-class ordering summary printed at the bottom of each
`recalibrated_verdict.md` is the textual headline that grounds the paper's
§4 prose.

### 6e. Workshop paper cross-link

For the surrounding argument — the closed-form bound that predicts
collapse $\to$ symmetric $\to$ asymmetric, how the empirical
`mixed` class sits between symmetric and asymmetric, and the
trainability-vs-NE-structure framing — see
[`paper/anvil_pub.bb-workshop.8/main.tex`](../paper/anvil_pub.bb-workshop.8/main.tex)
§4.

### 6f. k = 1 improvability oracle on the `no_convergence` cells (issue #428)

**Result**: A downstream consumer (issue #428, via the `rjwalters/thrust`
harness) reported that the $\kappa = 0.1$ / $c = 0.5$ `no_convergence`
cells are *degenerate for single-best-response learning* — i.e. that a
single agent facing 3 frozen uniform-random opponents has essentially no
improvable gap ($\leq +0.01\%$ team-return headroom, achieved by resting).
The repo-native k = 1 oracle does **not** reproduce this on the repo's
`minimal_specialization`-based phase-diagram cells: a deterministic
any-house firefighter beats the all-uniform baseline by **+14.4%** on
per-step team return (paired 95% CI on the delta $[+10.5, +16.6]$ per
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
oracle (as it is by NE conditional entropy, #430 Task 1); the
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

## 7. Trap-escape verdict rule for social-trap scenarios (rest_trap first)

**Problem**: `rest_trap`'s frozen NE (3×free_rider + 1×firefighter,
`bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`, team payoff
2984.04/episode ⇒ ≤ 248.67/step at min_nights = 12) sits *below* the
uniform-random baseline (302.87/step) — the equilibrium is team-suboptimal by
construction, so the `gap_closed` fraction ladder is degenerate there
(issue #434). Issue #436 adds the missing pieces: a measured scripted upper
anchor and a categorical verdict.

**Measured anchors** (per-step team reward, `rest_trap`, 4 agents):

| Anchor | Value | Provenance |
|---|---|---|
| NE per-step bound | ≤ 248.67 | frozen NE payoff / min_nights (upper bound; conservative for the "above NE" claim) |
| `always_rest` ×4 | 288.55 [285.20, 291.65] | scripted battery screen, n=2000 |
| uniform random | 302.87 (cited); 302.94 [301.46, 304.31] re-measured at n=10k — the 304.31 upper bound is committed as `random_ci95_hi` and anchors the `escaped_trap` rung | `SCENARIO_RANDOM_BASELINES` (#237); #436 final stage |
| `scripted_best` = `specialist` ×4 | **386.60 [386.17, 387.03]** | all-scripted team battery, n=10000, seed=0, host studio, commit `ee21e796` |

The battery (24 team profiles: uniform, always_rest, specialist, the
firefighter `{scope, work_prob}` family homogeneous + NE-shaped
k×firefighter+(4−k)×rest mixes + 16 random-search samples) decisively beats
random: the winning homogeneous `specialist` team gains **+83.67/step**
over uniform (paired 95% CI [+82.36, +84.89]). The trap structure is real —
both full resting (288.55) and the NE bound (≤ 248.67) score *below*
random play.

**Verdict rule** (`run_tier1_cell.classify_trap_verdict`, applied only when
`gap_source = "degenerate_reference"`): compute the percentile bootstrap 95%
CI `[lo, hi]` over **seeds** of the trained trailing-5 per-step team reward
(10,000 resamples, fixed RNG seed 436), then walk a nested one-sided ladder
on `lo`:

1. `lo > scripted_best.ci95_hi` → **`above_scripted_best`**
2. `lo > random_ci95_hi` (the random anchor's own measured 95% upper
   bound, 304.31 at n=10k; falls back to the `random` point when no
   measured bound is committed) → **`escaped_trap`**
3. `lo > ne_per_step_bound` → **`at_random`**
4. else → **`trapped_at_ne`**

Rung 2 deliberately anchors on the random baseline's *measured upper
bound*, not the bare point: the 302.87 point carries ±1.4/step measurement
noise at n=10k, so a sub-noise clearance of the point is not a
statistically supportable "above random" claim. This makes rung 2
symmetric with rung 1 (which anchors on `scripted_best.ci95_hi`) and with
the battery's own `beats_random` check.

If a scenario's scripted battery fails to beat random (documented #436
failure mode), `scripted_best` is recorded with provenance but rung 1 is
disabled and the rule operates on the NE + random anchors alone. The
anchors live in
`bucket_brigade.baselines.SCENARIO_GAP_REFERENCES["rest_trap"]`
(`ne_per_step_bound`, `random_ci95_hi`, `scripted_best`); `reference`
deliberately stays `None` so the fraction ladder remains off for
social-trap scenarios.

**het_ppo result (20 seeds, re-summarized without retraining)**: trailing-5
mean 306.26, seed-bootstrap CI [302.95, 309.33] → **`at_random`**. The CI
lower bound clears the 302.87 random *point* by 0.08/step, but that is
within the anchor's own measurement noise: it does not clear the n=10k
uniform re-measurement's upper bound (304.31), so het_ppo cannot be ruled
significantly above random. It does clear the NE bound (≤ 248.67) — PPO
does not fall into the resting trap — while sitting ≈ 80/step below
`scripted_best`. Honest reading: het_ppo is indistinguishable from random
play at the committed measurement precision and captures essentially none
of the measured scripted headroom on this scenario; the quantitative
headline remains `uplift_over_random = +3.39 ± 7.34`/step.

**Artifacts**:
- Battery script: [`experiments/p3_specialization/scripted_battery.py`](../experiments/p3_specialization/scripted_battery.py)
- Committed measurement: [`experiments/p3_specialization/scripted_battery/rest_trap.json`](../experiments/p3_specialization/scripted_battery/rest_trap.json), [`experiments/p3_specialization/scripted_battery/rest_trap.md`](../experiments/p3_specialization/scripted_battery/rest_trap.md)
- Verdict table with the `trap_verdict` column: [`experiments/p3_specialization/tier1_runs/tier1_verdict.md`](../experiments/p3_specialization/tier1_runs/tier1_verdict.md)

**Reproduce** (scripted policies only — no training; ~20 s locally):

```bash
uv run python experiments/p3_specialization/scripted_battery.py --scenario rest_trap
uv run python experiments/p3_specialization/run_tier1_cell.py \
    --trainer het_ppo --scenario rest_trap \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num-iterations 50 --summarize-only
uv run python experiments/p3_specialization/aggregate_tier1.py
```

**Source**: issue #436 (this rule), #434 / PR #438 (degenerate-reference
mechanism), PR #433 (het_ppo Phase 1 run), PR #432 (oracle pattern), #356
(why tier-1 trainers were expected to fail on rest_trap), #357 (paper
tracker).

---

## 8. het_ppo Phase 2 on the asymmetric_only cells: at_random

**Question**: does per-agent init asymmetry (`het_ppo`: disjoint per-agent
init streams, `--per-agent-init-seed-offset 1000`) let PPO learn the
asymmetric role structure that the NE demands on the two `asymmetric_only`
phase-diagram cells (`asym_b05_k09_c05` / `asym_b09_k09_c05`, β 0.5/0.9,
κ = 0.9, c = 0.5; frozen NE = 1×hero + 3×firefighter, team payoff 72.0095
**per episode** ⇒ ≤ +6.0/step at min_nights = 12)? This is the direct
interventional test issue #429 was designed around (Phase 1's rest_trap arm
turned out to be a degenerate social trap, §7).

**Protocol**: 20 seeds (42–61) × 50 iterations × 2048 rollout steps per
cell, host alc-9, train commit `8a532de1`; summaries regenerated locally via
`--summarize-only` (no retraining). The fraction ladder is off by design
(`ne_reference_per_episode_only`, PR #441): the NE anchor is per-episode,
gap metrics are per-step, so the story is `uplift_over_random` plus position
relative to the NE bound.

**Result — no. Both cells land at `at_random`** (per-step team reward;
random baseline −78.27/step, 95% CI [−83.88, −72.81], n=1000, PR #441):

| Cell | trailing-5 mean | uplift_over_random (±std, n=20) | 95% CI on mean uplift | trap verdict |
|---|---|---|---|---|
| `asym_b05_k09_c05` | −77.73 | +0.54 ± 8.74 | [−3.39, +4.47] | `at_random` |
| `asym_b09_k09_c05` | −75.65 | +2.62 ± 10.21 | [−1.98, +7.21] | `at_random` |

Both cells sit ~81–84/step below the NE per-step bound (+6.0); the best
single seed (b09 seed 46, trailing-5 ≈ −48/step) is still ~54/step short.
Per-agent init asymmetry injects behavioral differentiation at iteration 0
(within-seed action-entropy spread across agents = 1.28 nats) but the
spread does not significantly grow through training — trailing-5 spread is
1.59 ± 0.65 / 1.32 ± 0.70 nats, paired growth t = +1.62 / +0.20, both n.s.
(policy-entropy spread agrees: 0.29 → 0.36 ± 0.15 / 0.30 ± 0.16, t = +1.67
/ +0.27) — and pairwise MI declines over training (paired t = −2.5 /
−4.7): agents become more independent, not coordinated, no role structure
emerges beyond what the init streams mechanically injected, and no seed
converts that injected differentiation into a paying hero/firefighter
division of labor.

**Replication-pair caveat**: β is dynamically inert in bernoulli extinguish
mode, so the two cells are the same game — and the seed streams turned out
to be shared rather than scenario-hashed apart (iteration-0 returns match
exactly on 2/20 seeds, within 0.1% on 9/18 of the rest, within 0.5% on
13/20 overall; β's only live effect is as an
observation feature). Same-seed trailing-5 correlation r = +0.84; cell
means are statistically indistinguishable (Welch t = −0.67; same-seed diff
+2.07 ± 5.62/step, n.s.). Treat the pair as ~20 CRN-coupled draws of one
environment, **not** 40 independent seeds.

**NE-denominator caveat (#442)**: the 72.0095 registered NE payoff is
likely understated — the cross-β residual analysis
([`experiments/nash/phase_diagram/beta_residuals.md`](../experiments/nash/phase_diagram/beta_residuals.md),
PR #450) found `FF|hero|hero|FF` beats the registered `hero|FF|FF|FF` by
+9.55 ± 2.73/episode (seed-robust CRN re-evaluation, 55.36 vs 45.80), and
solver payoffs carry ~+26/episode winner's-curse bias vs CRN re-evaluation.
Moot for this verdict (trained policies are at random, far below either
candidate denominator), but any future gap fraction against 72.0095 would
be overstated; pending #445, report against both denominators.
**Resolution (#459 / #466)**: the exploitability audit
([`experiments/nash/phase_diagram/exploitability/RESULTS.md`](../experiments/nash/phase_diagram/exploitability/RESULTS.md))
confirmed both committed profiles are ε-NE at the repo-standard ε = 50 and
adopted `FF|hero|hero|FF` as the cells' NE anchor (winner's-curse-free CRN
team payoff 55.36 ± 3.44/episode ⇒ ≤ +4.7/step at min_nights = 12); the
`SCENARIO_GAP_REFERENCES` provenance and scenario registry now cite it.
The 72.0095 figure above is the historical solver record for the run as
executed. Still moot for this verdict.

**Artifacts**:
- Per-seed runs + cell summaries: [`experiments/p3_specialization/tier1_runs/het_ppo_asym_b05_k09_c05/`](../experiments/p3_specialization/tier1_runs/het_ppo_asym_b05_k09_c05/), [`experiments/p3_specialization/tier1_runs/het_ppo_asym_b09_k09_c05/`](../experiments/p3_specialization/tier1_runs/het_ppo_asym_b09_k09_c05/)
- Verdict table: [`experiments/p3_specialization/tier1_runs/tier1_verdict.md`](../experiments/p3_specialization/tier1_runs/tier1_verdict.md) (full caveats in the appended notes)

**Reproduce** (summaries only — no training; seconds locally):

```bash
uv run python experiments/p3_specialization/run_tier1_cell.py \
    --trainer het_ppo --scenario asym_b05_k09_c05 \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num-iterations 50 --summarize-only
uv run python experiments/p3_specialization/run_tier1_cell.py \
    --trainer het_ppo --scenario asym_b09_k09_c05 \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --num-iterations 50 --summarize-only
uv run python experiments/p3_specialization/aggregate_tier1.py
```

**Source**: issue #429 (het_ppo program; Phase 2 scope refresh comment),
#435 / PR #441 (frozen asym scenarios + measured random baseline), #442 /
PR #450 (NE-denominator caveat), PR #433 (Phase 1), #445 (seeded DO retry
that will settle the denominator).

---

## 9. Trap-escape ladder rung 1: budget scaling on rest_trap (4×/16×)

**Question** (issue #444, rung 1 of the trap-escape ladder): is the
rest_trap at-random plateau (§7) a training-budget artifact? het_ppo and
ippo re-run at 4× and 16× the tier-1 budget (200 / 800 iterations × 2048
rollout steps vs the standard 50 × 2048), 20 seeds (42–61) per cell, host
alc-2, train commit `ed0555af`. Budget is encoded in the artifact **root**
(`tier1_runs_trap_escape/{4x,16x}/`), not the cell name, so the standard
`--summarize-only` + `aggregate_tier1.py --tier1-root` pipeline applies
per root unchanged.

**Result — mostly no, with one marginal statistical escape.** Anchors as
in §7 (random point 302.87, measured 95% upper bound 304.31,
`scripted_best` 386.60 [386.17, 387.03]); values are trailing-5 per-step
team reward with seed-bootstrap 95% CIs:

| Budget | Trainer | trailing-5 mean | 95% CI | uplift_over_random (±std, n=20) | trap verdict |
|---|---|---|---|---|---|
| 1× (§7) | het_ppo | 306.26 | [302.95, 309.33] | +3.39 ± 7.34 | `at_random` |
| 4× | het_ppo | 307.66 | [304.03, 311.26] | +4.79 ± 8.36 | `at_random` |
| 4× | ippo | 306.13 | [301.77, 310.68] | +3.26 ± 10.24 | `at_random` |
| 16× | het_ppo | **307.83** | **[305.00, 310.71]** | +4.96 ± 6.59 | **`escaped_trap`** |
| 16× | ippo | 306.99 | [303.46, 310.39] | +4.12 ± 7.95 | `at_random` |

The 16× het_ppo cell is the first `escaped_trap` verdict recorded for any
trainer on rest_trap: its CI lower bound clears the random anchor's
measured 95% upper bound by +0.69/step, i.e. het_ppo at 16× budget is
statistically distinguishable from random play. Read it honestly:

- **No dose-response on the mean.** The plateau level is flat across
  budgets (306.26 → 307.66 → 307.83; paired same-seed contrasts: 4×−1× =
  +1.40 ± 4.79/step, t = +1.30; 16×−1× = +1.57 ± 4.80, t = +1.46; 16×−4×
  = +0.17 ± 4.25, t = +0.18; ippo 16×−4× = +0.86 ± 4.70, t = +0.82 — all
  n.s.). This extends §6b's flat-at-4× vanilla-PPO result to het_ppo,
  ippo, and 16×.
- **The verdict flip is variance-driven.** Extra budget tightens the seed
  spread (het_ppo uplift std 8.36 → 6.59), marching the CI *lower bound*
  up (302.95 → 304.03 → 305.00) around an unchanged mean: budget buys
  seed consistency, not performance.
- **The ~80/step learnability gap stands.** Mean uplift +4.96/step is
  ≈ 6% of the measured 83.7/step scripted headroom; the best single seed
  in all of rung 1 (4× het_ppo seed 61, trailing-5 325.84) captures ≈ 27%
  of it and remains ≈ 61/step below `scripted_best`. No seed comes close.
- **Multiplicity/CRN caveat**: the four cells share seed streams (seed 61
  is the top seed in 3 of 4 cells), so they are CRN-coupled rather than
  independent tests, and one of four 95%-CI tests crossing by 0.69/step
  is weak stand-alone evidence. The rule (#436/#440) is pre-registered,
  so the row stands as scored — as "marginally but significantly above
  random", not as closing the trap.

**Ladder decision**: #444's pre-registered stopping rule — any
`escaped_trap` ends the ladder — fires at rung 1. Recipe: het_ppo
(`--per-agent-init-seed-offset 1000`) on rest_trap, 800 iterations × 2048
rollout steps, 20 seeds. Rungs 2–4 (exploration variants, demonstration
seeding, PBT) are not run under this issue; the §7 hardness headline (no
gradient method captures more than a few percent of the scripted
headroom) is unchanged and, if anything, sharpened: 16× compute buys
statistical separation from random but no progress toward the specialist
solution.

**Artifacts**:
- Per-seed runs + cell summaries + per-root verdict tables:
  [`experiments/p3_specialization/tier1_runs_trap_escape/`](../experiments/p3_specialization/tier1_runs_trap_escape/)
  (see its README for layout; caveats in each root's
  `tier1_verdict_notes.md`)

**Reproduce** (summaries only — no training; seconds locally):

```bash
for spec in "4x:200" "16x:800"; do
  root=${spec%%:*}; iters=${spec##*:}
  for tr in het_ppo ippo; do
    uv run python experiments/p3_specialization/run_tier1_cell.py \
      --trainer $tr --scenario rest_trap \
      --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
      --num-iterations $iters --summarize-only \
      --output-root experiments/p3_specialization/tier1_runs_trap_escape/$root
  done
  uv run python experiments/p3_specialization/aggregate_tier1.py \
    --tier1-root experiments/p3_specialization/tier1_runs_trap_escape/$root
done
```

**Source**: issue #444 (ladder design + stopping rule; launch and results
comments), PR #440 / #436 (trap verdict rule + anchors), PR #438
(scenario-aware gap), §6b (4× vanilla-PPO precedent), #356 (why tier-1
trainers fail by construction).

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
