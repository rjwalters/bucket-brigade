# changelog: anvil_pub.bb-workshop.2 → .3

## Review source

- `paper/anvil_pub.bb-workshop.2.review/` — verdict: `advance: true`,
  35/44 (advance threshold met exactly), no critical flags, no
  blocker-severity comments. NeurIPS overlay 13/16 (informational).
- The three lowest-scoring dimensions in the v2 review were
  D6 (Figure quality, 3/4), D7 (Prose/structure, 3/4), and
  D9 (Rhetorical economy, 3/4). Per-dimension scores 1/3/5/8 are
  preserved (no regressions targeted; only the lowest three
  dimensions get load-bearing edits).

## Revision plan → resolutions

| Source                                        | Note                                                                                                | Resolution                                                                                                                                                                                                                  |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2.review (generic, major — single most important) | §4 β-independence overclaim: "confirming the β-independence prediction" not supported by n=2 cells per κ row with std ~0.33 ≫ 0.017 cell-to-cell diff. | §4 Results paragraph rewritten: "well within seed variance" retained, "confirming" replaced with "consistent with"; added one sentence naming the std-vs-difference asymmetry explicitly and a follow-up sentence routing the solver-level evidence to §3 (3/3 κ-rows identical across β). No change to Abstract (its β-independence mention is what the *reduction predicts*, not what the data confirms — already calibrated). No change to Conclusion (same reason). |
| 2.review (generic, minor — D6 Figure quality)    | Figure 2 (recalibrated_heatmap.tex) shows std as text label inside each cell; std ≈ mean on symmetric cells, signal-vs-noise obscured. | Deferred to `pub-figures`. The v2 review explicitly endorsed the matplotlib-replacement path for the camera-ready (overlay error-bar or Tufte sparkline). TODO comment added to `figures/recalibrated_heatmap.tex` header naming both options (a) drop-std-from-cell-text and (b) error-bar overlay, with the reviewer's option (b) endorsement recorded. The TikZ itself is unchanged in v3 because a partial fix (option (a) alone) loses information without the matplotlib swap landing. |
| 2.review (generic, minor — D7 Prose/structure)   | Page-fit margin tight at 3705 words; reviewer flagged minor redundancy in §3 closing, §5 triple "we do not claim", §6 Threats-to-analytical/empirical overlap, §3 Headline-finding restating biases. | Trimmed 51 words net (v2 3705 → v3 3654): §3 closing of (i)/(ii)/(iii) paragraph compressed (the "ring-Markov refinement; per-agent ownership" forward-reference removed because §6 carries the same content); §3 Headline-finding compressed ("non-trivially observable in the 7-cell preview" cut as it duplicates the (i)/(ii)/(iii) discussion; "the five sources of systematic bias" → "the full bias accounting"); §5 triple "we do not claim" collapsed to a single sentence ("Bucket Brigade does not scale to large populations, does not test emergent cooperation in any general sense, and is not a general-purpose MARL benchmark") preserving the forbidden-phrase negation context required by the BRIEF; §6 Threats-to-analytical cut the "if the full phase diagram turns out to be uninteresting" sentence which overlapped Threats-to-empirical's "full-grid sweep ... may shift absolute magnitudes" framing. |
| 2.review (generic, minor — D9 Rhetorical economy) | Same root cause as D7: redundancy and tight page-fit.                                              | Addressed via the D7 trim above. Same edits.                                                                                                                                                                                |
| 2.review (generic, minor — §1 orphan transition) | "We treat the analytical NE characterization as ground truth..." sentence is a structural orphan after the Contributions list. | Folded into the closing of Contribution (4) by prepending "Throughout, ..." — preserves the calibration framing while removing the standalone paragraph. Net: −1 line of standalone prose.                                  |
| 2.review (generic, minor — §4 methodological-obs paragraph framing) | "A methodological observation worth recording" softens the framing relative to the prior paragraph's "as a separate contribution." | Changed to "A methodological observation we report as a separate contribution" — matches the surrounding paragraph's framing and the Conclusion's framing.                                                                  |
| 2.review (generic, nit — `\#360-sweep` styling)  | Line 525 "\#360-sweep checkpoints" renders as "#360-sweep" which a reader unfamiliar with the project's tracker may misread. | Changed to "PPO-sweep checkpoints" — drops the issue number per reviewer suggestion. The in-repo `docs/PAPER_RESULTS.md` already names the issue numbers for readers who want them.                                          |

## Notes flagged as declined / deferred

| Source                                       | Note                                                                                                       | Resolution                                                                                                                                                                                                                  |
|----------------------------------------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2.review (generic, major — §1 Contributions list split) | Reviewer suggested splitting contribution (3) into 3a (protocol + first results) and 3b (per-cell baseline methodology) so the methodological inversion is a standalone contribution in the list. | Declined for v3. The Contributions list is already 4 items in a single paragraph; splitting (3) → (3a, 3b) would push to 5 items and risk page-fit. The per-cell-baseline-as-contribution framing is preserved in §4's methodological-observation paragraph and in the Conclusion. Re-evaluate at camera-ready if page budget allows. |
| 2.review (generic, major — §3 implicit-reader-test sentence) | Reviewer suggested adding a sentence defending "what is the value of the closed-form derivation if the thresholds are off by 10×" with the answer that it predicts which axis matters. | Declined for v3. The §3 Headline-finding paragraph already names the structural predictions (qualitative order, β-independence, collapse regime existence) as the load-bearing claim. Adding the reviewer's defensive sentence would add ~30 words against a tight page-fit budget; the implicit point is in the prose. Re-evaluate at camera-ready. |
| 2.review (generic, major — §4 sample-size language) | Reviewer suggested naming "We report ordering, not significance; the 4-seed-per-cell variance is larger than the class separations" explicitly. | Partial. The β-independence sentence rewrite (above) carries the std-vs-difference asymmetry. The Caveats paragraph already names n=2/2/3 cells per class. Naming "ordering, not significance" as a third sentence would add words; the reviewer's concern is substantively addressed by the β-independence rewrite. Re-evaluate at camera-ready. |
| 2.review (generic, major — compute disclosure) | "Single 16-core host in approximately two hours" lacks CPU model and memory footprint for reproducibility (NeurIPS overlay dim 5). | Declined for v3. The §6 Reproducibility paragraph names the source repo URL where the alc-2 host model is recorded in `experiments/REMOTE_EXECUTION.md`; the workshop reviewer-tier disclosure is acceptable. Re-evaluate before NeurIPS submission. |
| 2.review (generic, major — refs.bib internal-memo cites) | `bbenvspec` and `bbnestructure` are `@misc` internal-memo cites awaiting arXiv-preprint replacement. | Deferred to `pub-audit`. The arXiv preprint pathway is in flight; substitution is a one-line bib edit when the preprints land. Not the reviser's responsibility per `pub-revise.md` workflow. |
| 2.review (generic, minor — multiple other prose nits) | "right" in scare quotes (§1); phase-order parenthetical pointer (§2); Eq. (A) intersection empty interval (§3); 7-row truth table inline (§3); `gap_closed_homogeneous` one-more-sentence (§4); Table 1 NE column hedge (§5); HuggingFace status re-check (§6); Conclusion contributions re-listing (Conclusion); Fig 1 caption split (Fig 1); Fig 2 hatch+dagger (Fig 2 — already well-handled). | Declined for v3 — all are below the line for a workshop revision and would burn the trim budget. The Fig 2 hatch+dagger and the β=0.1 n/a cells were already noted as well-handled. Re-evaluate at camera-ready. |

## Sections left unchanged

- §2 (Environment): no changes. Reviewer's minor and nit comments (phase-order pointer; `$h_h$` notation) are below the trim threshold for a workshop revision.
- §5 Table 1: no changes. Reviewer's "NE characterisable? hedge" minor is judgment-call; the §3 honesty already qualifies the table.
- §6 Reproducibility paragraph (aside from `\#360-sweep` → `PPO-sweep`): no changes. HuggingFace "in progress" language is honest and the auditor's re-check is the right gate.
- §6 (a)/(b)/(c) defense paragraph: positively noted by reviewer; preserved verbatim.
- Conclusion: no changes. The β-independence mention there describes what the reduction predicts (correctly), not what the data confirms; the v2 phrasing is calibrated as-is.
- Abstract: no changes to the β-independence-related text. The "empirical agreement between predicted and PPO-realised verdicts" line refers to verdict-level evidence (which the reduction does predict correctly); the abstract's "$\beta$-independence at $c{=}0.5$" appears in a list of what the reduction "correctly predicts", not what the data confirms.

## Soft spots / qualifications carried forward

- The PPO sample size remains n=2/2/3 cells per NE class; the v3 hedge on β-independence makes this honest but does not patch the underlying evidence weakness. The full 75-cell grid (in progress) is the gate.
- The closed-form κ-thresholds remain off by 3–10×. The §3 prose owns this. No new analytical work in v3.
- Internal-memo cites (`bbenvspec`, `bbnestructure`) remain `@misc`. The arXiv replacement pathway is in flight but not landed at v3 cut.
- Figure 2 std-as-text remains; the matplotlib replacement is the right place for the visual fix and is queued for `pub-figures` and/or camera-ready.

## Files written

```
anvil_pub.bb-workshop.3/
  main.tex                          revised — §4 β-independence "confirming"
                                    → "consistent with" + std-vs-diff explanation;
                                    §1 orphan sentence folded into contribution (4);
                                    §3 Headline-finding compressed;
                                    §3 closing of (i)/(ii)/(iii) compressed;
                                    §4 methodological-observation framing
                                    aligned with surrounding paragraphs;
                                    §5 triple "we do not claim" compressed;
                                    §6 Threats-to-analytical compressed
                                    (overlap with Threats-to-empirical cut);
                                    §6 Reproducibility "\#360-sweep" → "PPO-sweep".
                                    Net: 3705 → 3654 words (-51 words).
  refs.bib                          carried over verbatim from .2/
                                    (no new citations needed; arXiv-preprint
                                    swap for `bbenvspec` / `bbnestructure`
                                    deferred to pub-audit).
  figures/phase_diagram.png         carried over verbatim from .2/
  figures/recalibrated_heatmap.tex  carried over from .2/ with a new TODO
                                    comment for pub-figures naming the
                                    matplotlib replacement options for D6.
                                    The TikZ visualization itself is
                                    unchanged in v3.
  changelog.md                      this file
  _progress.json                    iteration=3, max_iterations=4,
                                    revised_from=2
```
