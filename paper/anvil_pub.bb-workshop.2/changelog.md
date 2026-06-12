# changelog: anvil_pub.bb-workshop.1 → .2

## History note

The v1 draft was produced by `pub-draft` without a subsequent formal
`pub-review` pass — there are no `anvil_pub.bb-workshop.1.review/`,
`.audit/`, or `.litsearch/` sibling directories on disk. The v2 revision is
operator-driven and addresses a single dominant defect inherited from v1:
§4 (Trainability) was a labelled placeholder with no empirical data, and
the in-flight #360 PPO sweep has since completed and been re-aggregated
under a per-cell NE-anchored metric (#413). The changelog below maps each
operator-flagged revision item to its resolution, using the same shape the
formal critic loop would produce.

## Revision plan → resolutions

| Source                                             | Note                                                                                   | Resolution                                                                                                                                                                                                                                                |
|----------------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| operator (blocker)                                 | §4 was a labelled placeholder (`Data pending #360`); v1 had no empirical PPO numbers   | Replaced §4 wholesale with a real Results section reporting the 7-cell PPO sweep (4 seeds × 50 iter × 2048 rollout steps, `JointPPOTrainer`, ran on alc-2 in ~2h). Reports `gap_closed_ne` per cell from `recalibrated_verdict.md`.                       |
| operator (major)                                   | The headline metric needed to be per-cell-calibrated, not single-cell                  | Introduced `gap_closed_ne` in §4 with a methodology paragraph citing `bucket_brigade/baselines/per_cell.py` and `per_cell_baselines.json`. Framed the per-cell calibration as a separate methodological contribution.                                    |
| operator (major)                                   | §3 said "qualitative ordering should hold under proper calibration"; need confirmation | Added a sentence at the end of §3 (Headline finding paragraph) noting that the §4 trainability sweep bears out the qualitative ordering at the PPO-evaluation layer. Did not modify §3's existing claims about quantitative κ-threshold disagreement (those remain unchanged because the analytical thresholds are off independent of the calibration question). |
| operator (major)                                   | Honest reporting of the metric-inversion artefact under the old single-cell baseline   | Added a methodological-observation paragraph in §4 documenting that the OLD single-cell baseline reported the inverted ordering (asymmetric 0.262 > symmetric 0.091 > no_convergence -0.176), and that per-cell calibration resolves the inversion. Framed as a contribution rather than a flaw.                                                              |
| operator (major)                                   | Contributions list (§1) needed update — §4 now has data                                | Reworded contribution (3) from "protocol and falsifiable hypothesis with empirical results pending" to "protocol, per-cell NE-anchored metric, and first empirical results" with the ordering statement landed.                                          |
| operator (major)                                   | Figure 2 (PPO heatmap) placeholder needed real values                                  | Replaced `figures/ppo_heatmap_placeholder.tex` with `figures/recalibrated_heatmap.tex` carrying real per-cell `gap_closed_ne` values (mean ± std over 4 seeds), with hatched cells on the κ=0.1 row to flag the metric fallback to `gap_closed_homogeneous` where no NE policy exists. The pub-figures phase may replace this TikZ with a matplotlib render. |
| operator (minor)                                   | Abstract had a "results are pending" hedge                                             | Abstract rewritten: removed the "results are pending" hedge; reports the recovered ordering and the headline numbers (0.326, 0.106, near-zero) directly.                                                                                                |
| operator (minor)                                   | §1 had a "v2 will land data" sentence and a "data status" data hedge that was now stale | Removed the v1 "data status" paragraph at the top of §4 and the "v2 will land data" hedge in §1's analytical-hypothesis sentence. The new §4 prose is self-standing.                                                                                    |
| operator (minor)                                   | §6 reproducibility paragraph needed `per_cell.py` reference                            | Added a sentence to the §6 reproducibility paragraph naming `bucket_brigade/baselines/per_cell.py` and `per_cell_baselines.json` as the calibration code path. The earlier sentences (pip install, `docs/PAPER_RESULTS.md`, HuggingFace) are preserved. |
| operator (minor)                                   | §6 threats-to-empirical paragraph needed an update post-data                           | Rewrote the empirical-threats paragraph from "if ordering violates prediction, falsified" to "ordering recovered on 7 cells; full-grid sweep at longer budget may shift absolute magnitudes; per-cell baseline result is independent of the eventual full-grid outcome."                                                                                       |
| operator (minor)                                   | Conclusion needed update — §4 is no longer a hypothesis                                | Rewrote conclusion's empirical-hypothesis sentence to report the recovered ordering and the per-cell-baseline methodological contribution. The analytical-contribution sentence is unchanged.                                                              |

## Sections left unchanged

- §2 (Environment): no changes. Section scored well in the operator's mental review and the §4 revision does not depend on any env-spec wording.
- §5 (Related work): no changes. Spot-checked for "PPO success TBD" wording per BRIEF.md voice norms — clean. Phrases "emergent cooperation," "general-purpose MARL benchmark," and "scalable to large populations" remain only in the negation context that v1 had them in ("we do not claim ...").
- §6 (Discussion), aside from the two paragraph edits above: the limitations defence (a)/(b)/(c) and the analytical-threats paragraph are unchanged.

## Soft spots / qualifications carried forward

None. The new §4 numbers do not contradict any v1 claim that remains in the
paper:

- The §3 claim "qualitative phase order is recovered, quantitative thresholds disagree by 3–10×" is unchanged by the §4 data — the §4 results confirm the qualitative ordering and say nothing about quantitative κ-thresholds.
- The "predicted vs. observed thresholds" paragraph in §3 is about the
  equilibrium-solver verdict, not the PPO sweep; the new §4 data does not
  bear on it.

## Files written

```
anvil_pub.bb-workshop.2/
  main.tex                          revised (sections 1, 3 light touch, 4 fully replaced, 6 light touch, conclusion light touch; abstract rewritten)
  refs.bib                          carried over verbatim from .1/ (no new citations needed)
  figures/phase_diagram.png         carried over verbatim from .1/
  figures/recalibrated_heatmap.tex  NEW — replaces ppo_heatmap_placeholder.tex with real per-cell values
  changelog.md                      this file
  _progress.json                    iteration=2, max_iterations=4, revised_from=1
```
