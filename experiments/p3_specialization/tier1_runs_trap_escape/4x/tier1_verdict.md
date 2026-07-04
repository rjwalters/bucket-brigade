# Tier-1 sweep verdict

Verdict ladder: `gap_closed_mean >= 0.88` -> **closed**; `0.49 <= mean < 0.88` -> **partial_upper**; `0.20 <= mean < 0.49` -> **partial_lower**; `mean < 0.20` -> **insufficient**. Rows with a null `gap_closed` (**not_scored** / **not_scored_degenerate_reference**, #434) are never classified on the ladder; read their `uplift_over_random` (per-step, scenario scale) and the categorical `trap_verdict` (#436: seed-bootstrap 95% CI vs NE / random-upper-bound / scripted_best anchors -> `trapped_at_ne` / `at_random` / `escaped_trap` / `above_scripted_best`) instead.

| Trainer | Scenario | gap_closed (mean ± std) | uplift_over_random (mean ± std) | Trap verdict | n_seeds | Verdict |
|---------|----------|--------------------------|---------------------------------|--------------|---------|---------|
| het_ppo | rest_trap | n/a | +4.790 ± 8.363 | at_random | 20 ok | not_scored_degenerate_reference |
| ippo | rest_trap | n/a | +3.258 ± 10.244 | at_random | 20 ok | not_scored_degenerate_reference |

## Notes

### Trap-escape ladder rung 1 (#444), 4× budget root: both trainers `at_random`

This root is the 4×-budget arm of the #444 rung-1 budget-scaling test
(200 iterations × 2048 rollout steps vs the tier-1 standard 50 × 2048),
het_ppo + ippo on `rest_trap`, 20 seeds (42–61), host alc-2, train commit
`ed0555af`. Budget is encoded in the root path (`4x/`), not the cell
name, so the standard `run_tier1_cell.py --summarize-only` +
`aggregate_tier1.py --tier1-root` pipeline applies unchanged.

- **het_ppo: `at_random`** — trailing-5 mean 307.66/step, seed-bootstrap
  95% CI [304.03, 311.26]; the lower bound misses the random anchor's
  measured 95% upper bound (304.31) by 0.28/step.
- **ippo: `at_random`** — trailing-5 mean 306.13, CI [301.77, 310.68].

Paired against the committed 1× tier-1 cell
(`tier1_runs/het_ppo_rest_trap`, mean 306.26, same seeds), 4× buys
+1.40 ± 4.79/step (t = +1.30, n.s.): the mean plateau does not respond to
budget, extending `docs/PAPER_RESULTS.md` §6b's flat-at-4× vanilla-PPO
result to het_ppo and ippo on rest_trap. The best single seed in all of
rung 1 lives in this root (het_ppo seed 61, trailing-5 325.84/step) yet
still captures only ≈ 27% of the 83.7/step scripted headroom
(`scripted_best` = 386.60).

See the sibling `16x/tier1_verdict_notes.md` for the 16× arm, where
het_ppo's tighter seed spread produces rung 1's single (marginal)
`escaped_trap` verdict and the resulting ladder-stop decision.
