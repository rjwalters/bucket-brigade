# Tier-1 sweep verdict

Verdict ladder: `gap_closed_mean >= 0.88` -> **closed**; `0.49 <= mean < 0.88` -> **partial_upper**; `0.20 <= mean < 0.49` -> **partial_lower**; `mean < 0.20` -> **insufficient**. Rows with a null `gap_closed` (**not_scored** / **not_scored_degenerate_reference**, #434) are never classified on the ladder; read their `uplift_over_random` (per-step, scenario scale) and the categorical `trap_verdict` (#436: seed-bootstrap 95% CI vs NE / random-upper-bound / scripted_best anchors -> `trapped_at_ne` / `at_random` / `escaped_trap` / `above_scripted_best`) instead.

| Trainer | Scenario | gap_closed (mean ± std) | uplift_over_random (mean ± std) | Trap verdict | n_seeds | Verdict |
|---------|----------|--------------------------|---------------------------------|--------------|---------|---------|
| ippo | minimal_specialization | 0.125 ± 0.082 | +7.405 ± 4.862 | n/a | 3 ok | insufficient |
| influence | minimal_specialization | 0.120 ± 0.036 | +7.101 ± 2.121 | n/a | 3 ok | insufficient |
| hca | minimal_specialization | 0.081 ± 0.139 | +4.806 ± 8.268 | n/a | 3 ok | insufficient |
| lola | minimal_specialization | 0.005 ± 0.053 | +0.267 ± 3.132 | n/a | 3 ok | insufficient |
| het_ppo | rest_trap | n/a | +3.394 ± 7.340 | at_random | 20 ok | not_scored_degenerate_reference |
| het_ppo | asym_b09_k09_c05 | n/a | +2.616 ± 10.210 | at_random | 20 ok | not_scored_degenerate_reference |
| het_ppo | asym_b05_k09_c05 | n/a | +0.541 ± 8.736 | at_random | 20 ok | not_scored_degenerate_reference |

## Notes

### het_ppo / rest_trap (#429 / #434): recalibrated — no longer scored on the gap ladder

The vacuous `closed` verdict originally recorded for this row was a
baseline-scale artifact: the pre-#434 `run_tier1_cell.py::gap_closed` scored
**every** scenario against the `minimal_specialization` references, and on
rest_trap's reward scale uniform-random play alone mapped to
`gap_closed ≈ 6.58` (`iter0_gap_closed_mean = 6.573`), so the 0.49 gate was
passed by any policy, trained or not.

Issue #434 made `gap_closed` scenario-aware
(`bucket_brigade.baselines.SCENARIO_GAP_REFERENCES`) and this table was
regenerated from the committed per-seed `metrics.json` via
`run_tier1_cell.py --summarize-only` (no training re-runs). The het_ppo row
now reports:

- `gap_closed = null`, `gap_source = "degenerate_reference"`, verdict
  `not_scored_degenerate_reference` — the fraction ladder is **not
  applicable** on rest_trap. The frozen NE (`3×free_rider + 1×firefighter`,
  `team_payoff = 2984.04`/episode, i.e. ≤ 248.7/step at ≥ 12 nights;
  `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`) sits
  *below* the 302.87/step random baseline: rest_trap's equilibrium is
  team-suboptimal by construction (social trap), so no valid upper reference
  exists yet. (The NE payoff is per-episode while gap metrics are per-step —
  the ≤ 248.7/step bound suffices to establish degeneracy.)
- The honest scenario-scale headline is `uplift_over_random = +3.39 ± 7.34`
  /step (trailing-5 `306.26` vs random `302.87`, n=20 seeds) — marginal, not
  a rescue, consistent with #356's expectation that tier-1 trainers fail on
  rest_trap.
- The pre-#434 MINSPEC-scale value is preserved as the audit column
  `gap_closed_minspec_legacy_mean = 6.639`.

### minimal_specialization rows: gap values refreshed to the #416 specialist constant

The four `*_minimal_specialization` summaries committed in #346 were computed
at commit `734bec7e` with the historical `MINSPEC_SPECIALIST = -22.07`
(n=50); issue #416 re-derived the constant to `-28.38` (n=10k) but the
committed artifacts were never recomputed — there was no recompute tool until
`--summarize-only` (#434). The regeneration therefore shifted the minspec gap
fractions slightly (denominator 65.65 → 59.34; e.g. ippo `0.1128 → 0.1248`)
while the underlying per-seed `trailing5` team rewards and all four
`insufficient` verdicts are unchanged. The scenario-aware formula itself is
bit-identical to the historical MINSPEC formula for this scenario
(`SCENARIO_GAP_REFERENCES["minimal_specialization"]` = `(MINSPEC_RANDOM,
MINSPEC_SPECIALIST)`, drift-guarded in `tests/test_baselines_constants.py`).

### het_ppo Phase 2 (asymmetric_only phase-diagram cells): unblocked by #435

Phase 1 ran to completion (see recalibrated row above). Phase 2 was blocked
because its scenarios were not registered in `bucket_brigade/envs`. Issue
#435 resolved this — and found that the names this note previously listed
(`asym_b05_k05_c09`, `asym_b05_k09_c09`) came from a column-order misread of
`phase_diagram_table.md` (its columns are `c | β | κ`): the actual
`asymmetric_only` cells are (β=0.50, κ=0.90, c=0.50) and (β=0.90, κ=0.90,
c=0.50), now registered as `asym_b05_k09_c05` / `asym_b09_k09_c05` (frozen
IDs `-v1`). See `experiments/p3_specialization/het_ppo_runbook.md` and issue
#429.

### het_ppo Phase 2 (#429): asym_b05_k09_c05 / asym_b09_k09_c05 — `at_random`, no asymmetric role structure learned

Phase 2 ran the direct interventional test on the two `asymmetric_only`
phase-diagram cells whose frozen NE demands role asymmetry (1×hero +
3×firefighter, team payoff 72.0095 **per episode**; registered by #435/PR
#441). Protocol: 20 seeds × 50 iterations × 2048 rollout steps per cell,
`het_ppo` (per-agent init streams, `--per-agent-init-seed-offset 1000`), host
alc-9, train commit `8a532de1`; summaries regenerated locally via
`--summarize-only` (no retraining).

- **Ladder applicability**: `gap_closed = null`, verdict
  `not_scored_degenerate_reference`, reason `ne_reference_per_episode_only`.
  The registered upper anchor (72.0095/episode ≈ ≤ +6.0/step at ≥ 12 nights)
  is per-episode while the ladder is per-step, so
  `SCENARIO_GAP_REFERENCES` deliberately records no per-step upper reference
  (PR #441) and the fraction ladder is **not applied**. The quantitative
  story is `uplift_over_random` plus position relative to the NE bound.
- **Headline result — het_ppo does NOT learn the asymmetric NE role
  structure at this budget.** Trailing-5 per-step team reward vs the
  measured random baseline −78.27/step (95% CI [−83.88, −72.81], n=1000):
  - `asym_b05_k09_c05`: trailing-5 −77.73, `uplift_over_random = +0.54 ±
    8.74` (std, n=20; 95% CI on the mean [−3.39, +4.47]); trap verdict
    **`at_random`** (seed-bootstrap CI [−81.43, −73.82] does not clear the
    random anchor).
  - `asym_b09_k09_c05`: trailing-5 −75.65, `uplift_over_random = +2.62 ±
    10.21` (std, n=20; 95% CI on the mean [−1.98, +7.21]); trap verdict
    **`at_random`** (CI [−79.82, −70.84]).
  - Both cells sit ~81–84/step below the NE per-step bound (+6.0). Even the
    best single seed (b09 seed 46, trailing-5 ≈ −47.99/step) is ~54/step
    short of it. Paired uplift over each seed's own iteration-0 policy is
    also not significant (+3.09 ± 10.62, t = +1.30; +4.53 ± 12.18,
    t = +1.66). MINSPEC-scale audit values:
    `gap_closed_minspec_legacy_mean` = 0.168 / 0.203.
- **NE-denominator caveat (#442, REQUIRED)**: the 72.0095 registered NE is
  likely an *understated* denominator. The cross-β residual analysis
  (`experiments/nash/phase_diagram/beta_residuals.md`, PR #450) found the
  β=0.1 batch profile `FF|hero|hero|FF` beats the registered
  `hero|FF|FF|FF` by **+9.55 ± 2.73 per episode** (seed-robust, CRN
  re-evaluation 55.36 vs 45.80), and both solver payoffs in `results.json`
  carry ~+26/episode winner's-curse bias vs CRN re-evaluation. Any future
  gap fraction against 72.0095 would be *overstated*; pending the seeded DO
  retry (#445) report against both denominators. For this run the caveat is
  moot for the verdict — trained policies are at the random baseline, far
  below either candidate denominator.
- **Replication-pair finding — CRN-coupled replicas, not independent
  draws.** β is dynamically inert in bernoulli extinguish mode (the cells
  are the same game), and the per-seed streams turned out to be *shared*,
  not scenario-hashed apart: iteration-0 team rewards match exactly on 2/20
  seeds, within 0.1% on about half of the rest (9/18; 13/20 within 0.5%),
  because β's only live effect is as an
  observation feature (`bucket-brigade-core/src/engine/observation.rs`) that
  perturbs otherwise-identical policies. Same-seed trailing-5 correlation
  r = +0.84; trajectories diverge chaotically over training. Consistency
  holds: cell means are statistically indistinguishable (Welch t = −0.67 on
  trailing-5; same-seed diff b09−b05 = +2.07 ± 5.62/step, t ≈ +1.65, n.s.).
  Consequence: the pair provides ~20 paired draws of one environment, not
  40 independent seeds — do not pool them as n=40.
- **Role differentiation — injected at init, does not grow, no payoff.**
  Per-seed metrics carry per-agent policy/action entropies and pairwise
  MI/CMI (no per-agent rewards). The disjoint per-agent init streams
  mechanically inject behavioral differentiation at iteration 0
  (within-seed action-entropy spread across the 4 agents = 1.28 nats), and
  it does *not* significantly grow through training: trailing-5 spread is
  1.59 ± 0.65 (b05) / 1.32 ± 0.70 (b09) nats, paired growth t = +1.62 /
  +0.20, both n.s. (policy-entropy spread tells the same story: 0.29 at
  iter 0 → 0.36 ± 0.15 / 0.30 ± 0.16, t = +1.67 / +0.27, both n.s.).
  Meanwhile pairwise MI *declines* over training (paired Δ −0.22 ± 0.39,
  t = −2.5 on b05; −0.35 ± 0.33, t = −4.7 on b09): agents become more
  independent, not more coordinated, and no seed converts the injected
  differentiation into hero/firefighter division of labor that pays (team
  reward stays at random). Late-training CMI conditioner-degeneracy
  warnings (near-deterministic agents) corroborate collapse onto
  low-entropy but unproductive policies.

### het_ppo / rest_trap (#436): trap-escape verdict — `at_random`

Issue #436 added a measured upper anchor and a categorical four-way
trap-escape verdict for degenerate-reference (social-trap) rows, computed by
`run_tier1_cell.classify_trap_verdict` at re-summarization time
(`--summarize-only`, no retraining):

- **Anchors** (per-step team reward): NE bound ≤ 248.67 (frozen NE payoff
  2984.04/episode ÷ min_nights = 12), random 302.87
  (`SCENARIO_RANDOM_BASELINES`) with its own measured 95% upper bound
  304.31 (`random_ci95_hi`, the battery's final-stage n=10k uniform
  re-measurement 302.94 [301.46, 304.31]), and the measured `scripted_best`
  = 386.60 [386.17, 387.03] (homogeneous `specialist` team, n=10k, issue
  #436 Part A; artifact
  `experiments/p3_specialization/scripted_battery/rest_trap.{json,md}`).
- **Rule**: nested one-sided ladder on the lower bound of the seed-bootstrap
  95% CI of the trailing-5 per-step team reward (10k resamples, fixed seed):
  `lo > scripted_best_ci_hi` → `above_scripted_best`; `lo > random_ci95_hi`
  → `escaped_trap`; `lo > ne_bound` → `at_random`; else `trapped_at_ne`.
  The `escaped_trap` rung anchors on the random baseline's *measured 95%
  upper bound*, not the bare point (PR #440 review): the point carries
  ±1.4/step measurement noise at n=10k, so a sub-noise clearance of the
  point is not a statistically supportable "above random" claim. This makes
  rung 2 symmetric with rung 1 (`scripted_best.ci95_hi`) and with the
  battery's own `beats_random` check.
- **het_ppo result (20 seeds)**: trailing-5 mean 306.26, CI [302.95, 309.33]
  → **`at_random`**. The CI lower bound clears the 302.87 random *point* by
  only 0.08/step but sits below the random anchor's own n=10k measurement
  upper bound (304.31) — the clearance is within the anchor's measurement
  noise, so het_ppo cannot be ruled significantly above random. It does
  clear the NE bound (≤ 248.67), i.e. PPO does not fall into the resting
  trap, but it captures essentially none of the measured scripted headroom
  (Δ ≈ −80/step vs `scripted_best`). The quantitative headline stays
  `uplift_over_random = +3.39 ± 7.34`/step. The scripted battery also shows
  `always_rest` (288.55) and the NE bound (≤ 248.67) are both *below*
  random: the trap is real, and a fully-scripted specialist team
  demonstrates the scenario is not reward-capped at random-level play.
