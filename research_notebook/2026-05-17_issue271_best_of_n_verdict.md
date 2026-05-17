# Issue #271 — Random-init MLP best-of-N verdict

**Date:** 2026-05-17
**Scenario:** `minimal_specialization`
**Sweep:** 1000 random-init policy networks × 20 episodes (phase 1), then top-1% × **200 episodes (K=200 stability re-eval)** (phase 2). Both `independent` and `shared` init protocols.

## Verdict — both protocols: `random_play_basin`

The K=200 stability re-eval rejects the active-anti-attractor hypothesis. PPO is **not** doing worse than naive random search.

| Metric | Independent | Shared |
|---|---:|---:|
| Population mean (n=1000, ep=20) | -90.21 | -90.47 |
| Population std | 20.31 | 20.34 |
| Phase-1 best (lucky 20-ep draw) | gap_closed ≈ 0.70 | gap_closed ≈ 0.71 |
| **Phase-2 top-10 (K=200 re-eval)** | **gap_closed ≈ 0.0** | **gap_closed ≈ 0.0** |
| Phase-2 single best seed | gap_closed = 0.122 | gap_closed = 0.146 |
| Phase-1 → phase-2 drift (mean) | -46.22 | -46.34 |

## What this means

The earlier excitement on the phase-1 best random net (-32.51, gap_closed ≈ 0.70) was pure 20-episode sampling noise. The curator's mandatory K=200 re-eval at top-1% was specifically designed to catch this kind of false positive, and it did. The phase-1 → phase-2 drift of -46 points is enormous: phase-1 top picks are essentially indistinguishable from typical random nets when evaluated with adequate statistical power.

**There are no random-init policy networks that materially outperform PPO's converged plateau on `minimal_specialization`.**

## Thesis implications

The two competing readings the experiment was designed to discriminate:

| Hypothesis | Status |
|---|---|
| **Active anti-attractor** (PPO < random search) | ❌ Falsified by this experiment |
| **Basin trap** (PPO finds same locale as random init) | ✅ Confirmed |

PPO is not being pulled *away* from cooperative basins; it is *locally trapped* near random play. The cooperative basin is unreachable via local gradient updates from random init, but there's no evidence (from this experiment) that the gradient direction is actively misaligned — it's that local updates can't *find* the cooperative basin.

This weakens the strongest form of the misaligned-gradient thesis ("active anti-cooperation gradient") and confirms a more conventional form ("local minima in cooperative MARL"). The Goodhart-in-RL framing still applies, but at the level of "gradient too local" rather than "gradient pointed wrong."

## What still needs to be tested

The basin-trap reading predicts that **the cooperative basin IS stable under PPO updates**, but PPO can't reach it from random init. The discriminating experiment is **#270 (BC-init then PPO)**:

- BC reaches `gap_closed = 0.934` (per PR #278 cross-finding). The specialist policy is representable in the architecture.
- If PPO continuation from that init holds → basin trap fully confirmed (specialist basin is reachable AND stable, just not findable from random)
- If PPO continuation collapses → an unexpected third hypothesis: specialist basin isn't even locally stable. That'd be a deeper Goodhart-shaped failure than this experiment detected.

#270's PPO continuation is currently CPU-starved on the sibling sweep but expected to land soon.

## Artifacts

All under `experiments/p3_specialization/diagnostics/results/issue271_random_mlp_search/`:
- `results_independent.json` / `results_shared.json` — full 1000-seed phase-1 data
- `top_candidates_independent.json` / `top_candidates_shared.json` — sorted top-1%
- `stability_independent.json` / `stability_shared.json` — K=200 phase-2 re-eval (the discriminator)
- `distribution_independent.png` / `distribution_shared.png` — visualizations
- `summary.md` — full raw verdict report

## Recommended re-prioritization of the queue

| Issue | Previous priority | New priority post-#271 |
|---|---|---|
| **#270 BC-init then PPO** | Phase 1, high | Top — load-bearing for basin-trap confirmation |
| **#288 PBT with mutation** | Phase 3 | Promoted — direct test of basin escape |
| **#291 single-agent long-horizon** | Phase 2 | Stays high — paper-scope-determining |
| #287 LOLA, #284 COMA, #283 potential shaping | Phase 2 (load-bearing) | Demoted — fixes gradient *direction*, which isn't the problem |
| **#292 minimal 2x2 toy reproduction** | Filed | Promoted — basin trap is a cleaner story to demo |

## References

- `research_notebook/2026-05-17_thesis_misaligned_gradients.md` (gets thesis-pivot update)
- `research_notebook/2026-05-17_lit_review.md`
- PR #277 (this PR's infrastructure)
- Sister #270 (the next-most-important pending experiment)
