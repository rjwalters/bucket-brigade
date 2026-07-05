## Notes

### Trap-escape ladder rung 1 (#444), 16× budget root: het_ppo `escaped_trap` — statistically above random, but marginal

This root is the 16×-budget arm of the #444 rung-1 budget-scaling test
(800 iterations × 2048 rollout steps vs the tier-1 standard 50 × 2048),
het_ppo + ippo on `rest_trap`, 20 seeds (42–61), host alc-2, train commit
`ed0555af`. Budget is encoded in the root path (`16x/`), not the cell
name, so the standard `run_tier1_cell.py --summarize-only` +
`aggregate_tier1.py --tier1-root` pipeline applies unchanged (cell dirs
keep the canonical `<trainer>_<scenario>` naming).

- **het_ppo: `escaped_trap`** — trailing-5 mean 307.83/step,
  seed-bootstrap 95% CI [305.00, 310.71]; the CI lower bound clears the
  random anchor's measured 95% upper bound (304.31) by +0.69/step. This is
  the first `escaped_trap` verdict recorded for any trainer on rest_trap:
  het_ppo at 16× budget is statistically distinguishable from random play.
- **ippo: `at_random`** — trailing-5 mean 306.99, CI [303.46, 310.39];
  lower bound misses the 304.31 anchor by 0.85/step.

**Read the escape honestly — it is statistical, not substantive:**

- The mean uplift over random is **+4.96 ± 6.59/step**, ≈ 6% of the
  measured 83.7/step scripted headroom (`scripted_best` = 386.60 [386.17,
  387.03]). No trained policy team is anywhere near the scripted
  specialist solution; the ~80/step learnability gap remains open.
- **The verdict flip vs 4× is variance-driven, not level-driven.** The 4×
  het_ppo cell has an indistinguishable mean (307.66 vs 307.83; paired
  16×−4× per-seed diff +0.17 ± 4.25/step, t = +0.18, n.s.) but a wider
  seed spread (uplift std 8.36 vs 6.59), so its CI lower bound (304.03)
  misses the anchor by 0.28/step. Extra budget makes seeds more
  *consistent* around the same plateau; it does not raise the plateau.
- **No dose-response on the mean.** Paired against the committed 1×
  tier-1 cell (`tier1_runs/het_ppo_rest_trap`, mean 306.26, CI [302.95,
  309.33], same seeds): 4×−1× = +1.40 ± 4.79/step (t = +1.30, n.s.),
  16×−1× = +1.57 ± 4.80/step (t = +1.46, n.s.). ippo 16×−4× = +0.86 ±
  4.70/step (t = +0.82, n.s.). What rises with budget is the CI *lower
  bound* (302.95 → 304.03 → 305.00 for het_ppo), i.e. reliability.
- **No individual seed reaches toward `scripted_best`.** Best 16× seed is
  het_ppo seed 61 at 321.59/step; best anywhere in rung 1 is the 4×
  het_ppo seed 61 at 325.84/step — +22.97/step over the random point,
  ≈ 27% of the headroom, still ≈ 61/step short of 386.60.
- **CRN coupling / multiplicity caveat.** All four rung-1 cells share the
  same seed streams (seed 61 is the best seed in 3 of 4 cells; seeds
  47/49 the worst), so the cells are CRN-coupled rather than independent
  replications, and one of four 95%-CI tests crossing an anchor by
  0.69/step is weak evidence on its own. The verdict rule is
  pre-registered (#436/#440), so the row stands as scored, but the escape
  should be described as "marginally but significantly above random", not
  as closing the trap.

**Provenance / rebuild note**: the 16× cells were trained as two
concurrent 10-seed halves (seeds 42–51 and 52–61) writing into the same
cell directories; each half wrote a partial 10-seed `cell_summary.json`
(last writer wins). The committed summaries were rebuilt from all 20
`seed_*/metrics.json` via `--summarize-only` (byte-stability verified by
running the rebuild twice). Launch logs stay on alc-2
(`tmux trap-escape-rung1`; logs `<root>/{het_ppo,ippo}_16x_{a,b}.log`).

**Ladder decision (#444)**: the pre-registered stopping rule is "any
`escaped_trap` → that is the rung-1 result; stop the ladder." Rung 1
therefore terminates the ladder with the recipe: **het_ppo
(`--per-agent-init-seed-offset 1000`) on rest_trap at 800 iterations ×
2048 rollout steps, 20 seeds** — with the explicit caveat that the escape
clears the random anchor by ~5/step while leaving ~79/step of measured
scripted headroom untouched (the environment-hardness headline of
`docs/PAPER_RESULTS.md` §7 stands).
