## Notes

### het_ppo / rest_trap (#429): `gap_closed` values > 1 are a baseline-scale artifact

`run_tier1_cell.py::gap_closed` scores **every** scenario against the
`minimal_specialization` references (`MINSPEC_RANDOM = -87.72`,
`MINSPEC_SPECIALIST = -28.38`, per-step team reward). `rest_trap` lives on a
completely different reward scale — uniform-random play alone scores
`302.87`/step there (`SCENARIO_RANDOM_BASELINES["rest_trap"]`) — so for the
`het_ppo`/`rest_trap` row the fraction is far outside `[0, 1]` and the verdict
ladder is not meaningful:

- Uniform-random play on `rest_trap` maps to `gap_closed ≈ 6.58`; the 0.49
  gate maps to a per-step team reward of `-58.6`, which even untrained
  iteration-0 policies clear trivially (`iter0_gap_closed_mean = 6.573`).
- **Formal gate result:** `gap_closed_mean = 6.639 ± 0.124` (n=20 seeds)
  `>= 0.49` → verdict `closed`, gate **passed as computed by the committed
  pipeline** — but the pass is vacuous on this scenario, because random play
  also passes.
- **On rest_trap's own scale:** trailing-5 per-step team reward
  `306.26 ± 7.53` vs `302.87` for uniform random and `302.35 ± 8.16` at
  iteration 0. The training uplift is `+3.91 ± 8.85`/step (paired per-seed,
  t ≈ 1.98, n=20) — marginal, not a rescue.
- A rest_trap-referenced gap fraction is **not currently computable**: the
  frozen rest_trap NE (`3×free_rider + 1×firefighter`,
  `team_payoff = 2984.04`/episode, i.e. ≤ 248.7/step at ≥ 12 nights;
  `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`) sits
  *below* the random-play baseline, so a `(trained − random)/(NE − random)`
  denominator is negative/degenerate. rest_trap's equilibrium is
  team-suboptimal by construction (social trap).
- This is the same stale-reference problem #413 fixed for
  `run_phase_diagram_ppo.py` via per-cell baselines; `run_tier1_cell.py`
  still applies the MINSPEC constants to all scenarios. The
  `minimal_specialization` rows are unaffected (their scale *is* the MINSPEC
  scale).

### het_ppo Phase 2 (asymmetric_only phase-diagram cells): blocked

Phase 1 passed the formal gate, but the Phase 2 scenarios
(`asym_b05_k05_c09`, `asym_b05_k09_c09`) are not registered in
`bucket_brigade/envs` — they exist only in the runbook/launcher docs. Phase 2
is blocked on the #358 cell-to-scenario follow-up (see
`experiments/p3_specialization/het_ppo_runbook.md` and issue #429).
