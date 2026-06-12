# changelog: anvil_pub.bb-workshop.3 → .4

## History note

v4 is a **content revision** driven by four new sweeps that landed today,
not a critic-loop revision. The v3.audit verdict was `advance: true` (35/44,
no critical flags, no blocker comments). v4 carries forward all v3 audit
conclusions and adds the new findings; nothing in the v3 review or audit
asked for the v4 changes — they are surfaced by the new data alone.

Sweeps that landed today (2026-06-11):

- **Phase 1 (Nash):** phase-diagram grid expanded from 7 → **39 cells**
  (3×5×3 grid: β ∈ {0.1, 0.5, 0.9} × κ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × c ∈
  {0.5, 1.0, 2.0}, less 6 cells in the high-κ × c=0.5 corner that are
  subsumed by the c=1.0 row). New verdict tally:
  symmetric_only=12, asymmetric_only=11, **mixed=10** (NEW class),
  no_convergence=6.
  Source: `experiments/nash/phase_diagram/results.json`.
- **Phase 2a/b:** per-cell Random/Specialist baselines refreshed for the
  wider grid (38 cells); conditional-entropy panel (31 cells, not paper-
  load-bearing in v4 but kept for the appendix).
  Sources: `per_cell_baselines.json`, `conditional_entropy.{json,csv}`.
- **Phase 2c (PPO, original budget):** 37 of the 39 Nash cells × 4 seeds at
  50 iter × 2048 rollout steps with `JointPPOTrainer`. Two cells
  (b=0.10, k=0.50, c=0.50 and b=0.10, k=0.90, c=0.50) were skipped by the
  sweep schedule. ~6 h wall on alc-2.
  Source:
  `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.{json,md}`.
- **Phase 2d (PPO, 4× budget on no_convergence):** 6 no_convergence cells ×
  4 seeds at 200 iter × 4096 rollout steps. ~6 h wall on alc-6.
  Source:
  `experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.{json,md}`.

## Headline numbers v4 reports (changed from v3)

**Per-class `gap_closed_ne` (original budget, n cells per class):**

| NE class | n | gap_closed_homogeneous | gap_closed_ne |
|---|---:|---:|---:|
| symmetric_only | 11 | +0.051 | +0.180 |
| **mixed (new)** | 9 | +0.050 | +0.107 |
| asymmetric_only | 11 | +0.033 | +0.059 |
| no_convergence | 6 | −0.049 | n/a (fallback) |

Ordering under both `gap_closed_homogeneous` and `gap_closed_ne`:
symmetric > mixed > asymmetric > no_convergence, matching the analytical
prediction. **The v3 7-cell ordering was `symmetric (0.326) > asymmetric
(0.106) > collapse (−0.033)`**; the wider grid both confirms the order and
flattens the absolute magnitudes by ~0.15 on symmetric/asymmetric cells
(more cells, more cell-to-cell heterogeneity within class).

**4× budget on no_convergence (n=6):** `gap_closed_homogeneous` worsens from
−0.049 (original budget) to −0.108 (4× budget). PPO failure on
no_convergence cells is structural, not budget-limited.

**β-independence at the wider grid:** every (κ, c) row with ≥2 β samples
(13 such rows) shows identical NE verdict across all sampled β. v3 hedged
this as "consistent with, does not establish" because the 7-cell preview
had n=2 cells per κ row; v4's 13 rows give a much wider test surface and
v4 reports the verdict-level β-independence as held outright while still
noting the metric-level cross-β range is comparable to per-seed std (so
PPO does not independently sharpen the test beyond the solver's evidence).

## Revision plan → resolutions

| Source        | Note                                                                                                  | Resolution                                                                                                                                                                                                          |
|---------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| new data (Phase 1)  | Grid 7 → 39 cells; new 4th NE class `mixed` (10/39 Nash cells, 9/37 PPO cells).                | §3 prose updated: phase diagram now reports four empirical classes (n=12/11/10/6). Added a new paragraph noting the closed-form bound `tilde_A * kappa * (1-kappa)^k vs c_gap` cannot predict the `mixed` boundary at higher c — surfaced as an honest hedge, not as a fourth analytical regime. §1 contributions list updated to mention the 4-class taxonomy. Abstract phase-order updated to `collapse → symmetric → mixed → asymmetric`. |
| new data (Phase 2c) | 37-cell PPO sweep at original budget; new headline `gap_closed_ne` numbers per class.          | §4 Protocol paragraph: 37 cells (was 7), 148 runs (was 28), 6 h wall on alc-2 (was 2 h). §4 Results paragraph: 4-class ordering 0.180 > 0.107 > 0.059 > −0.049 reported with n per class. Figure 2 caption updated for the 3×5×3 grid. The data source path used by the figure script changed from `phase_diagram_ppo/recalibrated_verdict.json` to `phase_diagram_ppo_v2/...`. |
| new data (Phase 2d) | 4×-budget sweep on no_convergence cells.                                                       | NEW §4 "Is PPO failure on no-convergence cells budget-limited?" paragraph reporting `gap_closed_homogeneous` shift from −0.049 → −0.108. Frames as: PPO failure is structural, not budget-limited. §1 contribution (3) updated to mention this; abstract updated; conclusion updated.        |
| new data (β-independence) | 13/13 rows show identical verdict across β in the wider grid; v3 hedged "consistent with."   | §3 prose: "every (κ, c) row shows an identical verdict across all sampled β (13/13 rows with ≥2 β samples)." Upgraded from v3's "3/3 κ-rows." §4 metric-level hedge retained: per-cell std still comparable to cross-β metric range. |
| new data (Figure 2) | 7-cell 3×3 grid no longer adequate; need 3-panel 3×5 layout.                                   | Rewrote `figures/src/recalibrated_heatmap.py` to a 3-panel layout (one panel per c ∈ {0.5, 1.0, 2.0}), each 3×5 cells. Generated `figures/recalibrated_heatmap.pdf`. Tufte-style horizontal std bars preserved from v3. n/a cells (b=0.10, k=0.50, c=0.50 and b=0.10, k=0.90, c=0.50) shown hatched grey. Hatching+dagger preserved for no_convergence cells. Figure caption updated. |
| new data (sample size disclosure) | v3 used n=2/2/3 cells per class; v4 has n=11/9/11/6.                                   | §4 Caveats and §6 Threats paragraphs updated to reflect the new n values. The "ordering, not significance" framing carries forward. "n=2 per κ row" v3 wording for β-independence retired; v4 reports the wider per-class statistics. |
| new data (analytical disclosure) | v3 said the 3-10× threshold gap was unproven outside 7 cells; v4 confirms it on 39 cells. | §3 prose: "The derivation gets the phase order right on the 39-cell grid but the thresholds wrong by 3–10×, the same gap we owned at the 7-cell preview." Claim preserved, scope tightened to the new grid. |

## What was NOT changed (preserved verbatim or near-verbatim from v3)

- §2 (Environment): no content changes. The minimal contract is still
  correct at the wider grid.
- §3 closed-form derivation (Reduction, eq. S/A/C, Predicted thresholds
  paragraph): unchanged. The mean-field algebra is independent of the grid
  width; only the "Predicted vs. observed" paragraph and the Headline-
  finding paragraph were edited to incorporate the wider-grid evidence and
  the new `mixed` class.
- §5 Related work: unchanged. The comparison table's claims about Bucket
  Brigade are still true at the wider scale (`mixed within cell` was
  already in the table; the wider grid sharpens but does not change it).
- §6 (a)/(b)/(c) defense paragraph: unchanged.
- §6 Reproducibility paragraph: unchanged (still names
  `bucket_brigade/baselines/per_cell.py` and
  `experiments/nash/phase_diagram/per_cell_baselines.json`).
- refs.bib: unchanged. No new citations needed for v4.
- figures/phase_diagram.png: unchanged. The PNG is the c=0.5 panel of the
  Nash phase diagram, which is still valid; the figure caption is updated
  to acknowledge the wider grid and the new `mixed` class without changing
  the image.

## Soft spots / qualifications carried forward

- The closed-form κ-thresholds remain off by 3–10× on the wider grid (the
  v3 claim survives the wider test). §3 owns this.
- The closed-form bound does not predict the empirical `mixed` class.
  v4 surfaces this as an honest hedge, not as a fourth analytical regime.
  Tightening the reduction to predict the `mixed` boundary is named in
  §6 as the natural extension.
- Internal-memo cites (`bbenvspec`, `bbnestructure`) remain `@misc`. The
  arXiv replacement pathway is in flight but not landed at v4 cut. Will
  be handled in pub-audit.
- 4×-budget sweep only runs on no_convergence (n=6); the cross-class
  longer-budget question is not answered for symmetric/mixed/asymmetric.
  §4 and §6 disclose this honestly.
- HuggingFace baselines pathway: still "in progress" at v4 cut.
- The 2 cells skipped in the PPO sweep (b=0.10, k=0.50, c=0.50 and
  b=0.10, k=0.90, c=0.50) are disclosed in §4 protocol; the figure
  marks them n/a.

## Files written

```
anvil_pub.bb-workshop.4/
  main.tex                          new — §1 contributions list updated;
                                    §3 Predicted-vs-observed and Headline-
                                    finding paragraphs expanded for the
                                    39-cell grid and the new `mixed` class;
                                    §4 Protocol updated (37 cells, 4 seeds,
                                    alc-2, 6 h); §4 Results expanded for
                                    4-class ordering; NEW §4 paragraph on
                                    4× budget; §4 Caveats and §6 Threats
                                    updated for new sample sizes; Figure 1
                                    caption updated for wider grid; Figure 2
                                    caption updated for 3-panel 3×5 layout;
                                    Abstract updated; Conclusion updated.
                                    Word count: ~4000 (v3 was 3654).
  refs.bib                          carried over verbatim from .3/ (no new
                                    citations needed).
  figures/phase_diagram.png         carried over verbatim from .3/.
  figures/recalibrated_heatmap.pdf  NEW — generated by the v4 figure script;
                                    3-panel 3×5 layout, one panel per c.
                                    Tufte-style horizontal std bars
                                    preserved from v3. n/a cells and
                                    no_convergence hatching preserved.
  figures/src/recalibrated_heatmap.py  rewritten — reads from
                                    `phase_diagram_ppo_v2/recalibrated_verdict.json`
                                    (was `phase_diagram_ppo/...`),
                                    handles 3-panel layout, treats `mixed`
                                    as an NE-bearing class (uses
                                    gap_closed_ne, not the fallback).
  changelog.md                      this file
  _progress.json                    iteration=4, max_iterations=4,
                                    revised_from=3
```

## Notes for next stage

This is iteration 4 of max_iterations=4. If a v5 is needed (e.g., to land
the cross-class longer-budget sweep or the arXiv-preprint cite swap), the
next revisor will need to bump `max_iterations` in `.anvil.json` before
proceeding.

Figure 2 was regenerated for the 37-cell grid in this revision (run via
`uv run --with matplotlib python figures/src/recalibrated_heatmap.py`).
No `pub-figures` follow-up is needed for the headline figure.
