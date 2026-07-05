# Findings — anvil_pub.bb-workshop.9 (cross-section observations)

## Review-context notes

- **Prior review sibling**: the spec's prior-review lookup targets
  `anvil_pub.bb-workshop.8.review/`, which does not exist (v7-v9 were
  revised outside the anvil lifecycle). Handled per spec as the
  absent-prior case: `prior_rubric_id = None`, no `prior_rubric_id`
  field in `_summary.md`, and no rubric-version-transition subsection
  (the most recent existing review, v6, was also scored against
  `anvil-pub-v2`, so no transition occurred in any case).
- **Render-gate**: skipped fail-open — no `paper.pdf` /
  `compile-log.txt` from a v9 `pub-audit` exists yet
  (`pub-review: render-gate skipped — paper.pdf / compile-log.txt not
  present; run pub-audit first`). An independent build check was run
  instead: full `pdflatex + bibtex + pdflatex x2` with the shipped
  `anvil-paper.cls` exits 0, 26 pages, zero unresolved
  references/citations in the final pass, byte-count-consistent with
  the committed `main.pdf` (26 pages).
- **Numeric-consistency pre-check (issue #462)**: run with
  `--write-review`; sidecar at `anvil_pub.bb-workshop.9.numeric/`;
  1043 numbers extracted, 1 claim checked, 0 findings.
- **Venue overlay**: `.anvil.json` declares `venue: "neurips"`;
  `discover_venue_rubric` resolved `anvil-pub-neurips-v1` (/16,
  advisory) from `.anvil/skills/pub/rubrics/neurips.yaml`; overlay
  scored 11/16 in `_review.venue.json`.
- **Corpus / subject-voice tiers**: inactive (no `corpus:` or
  `subjects` declarations in the project layout) — no provenance
  back-check, no subject-voice sub-pass, per the silence-when-absent
  convention.

## Cross-section observations

1. **The negative-results arc is the paper's best property and is
   consistently executed.** The entropy-predictor retirement (§4), the
   beta-inertness correction (§3/§4/App. B — "exact by construction",
   9 sites checked, all consistent), the k* binary-threshold
   falsification (§4, pre-registered, exact + permutation tests), and
   the post-hoc/kappa-confounded flagging of the surviving k*=k_max
   observation at every one of its five mentions (verified by grep, as
   the v9 stale-claims audit also records) form a coherent
   falsification-first narrative. Rigor and honesty are not the
   problem anywhere in this paper.
2. **Artifact traceability is genuinely excellent, which makes the one
   drift sharper.** All 16 spot-checked `experiments/...`,
   `bucket_brigade/...`, and `docs/...` paths exist; the §5 anchor
   ladder numbers (2984.04/ep, 386.60, 302.87, 304.31, 306.26, 307.83,
   305.00, 288.55) all verify against their committed artifacts. The
   single exception — `recalibrated_verdict.json` regenerated with
   n=20 data after the paper's numbers were frozen — is therefore the
   highest-leverage fix (majors, priority 1).
3. **Presentation debt accumulated over the three unreviewed
   revisions.** v6 closed its punch list cleanly (40/44); v7-v9 added
   large amounts of load-bearing content without a critic pass, and
   the deductions in this review (overfull hboxes from new artifact
   paths, abstract growth, caption errors surviving from earlier
   versions, duplicated caveats) are exactly the class of defect the
   review loop exists to catch. None requires re-running an
   experiment.
4. **The 12/12 equilibrium-coverage closure (§5) is well-supported.**
   The seeded symmetric-DO retry, the 407.5/episode exploitability
   bound on the final mixture, and the standing asymmetric NE are all
   cited to `experiments/nash/rest_trap_seeded_do/RESULTS.md` (exists)
   and the framing ("a property of the game rather than an open hole")
   is earned by the battery-seeded design.

## Disposition of v6 review concerns across v7-v9

The most recent review sibling is `anvil_pub.bb-workshop.6.review/`
(40/44, advance). Its open items, checked against v9:

| v6 item | Status in v9 |
|---|---|
| Dim 9: appendix bulk / promote to supplementary | **Not addressed** (appendices unchanged; body grew further) |
| App. A §A.6 named templates without cites | **Not addressed** |
| §B.4/§B.5 boundary layout accident | **Not addressed** |
| §B.2 small-q(k) Taylor step not named as bias | **Not addressed** (B.5 still says "Five sources") |
| HuggingFace baselines "in progress" | **Not addressed** |
| Fig. 1 caption panel-layout half-sentence (nit) | **Not addressed** |
| Fig. 1 content + Table 2 overflow (v5 majors, closed in v6) | **Still closed** — no regression |

All v6 items were explicitly non-blocking camera-ready polish; their
persistence is noted for the reviser's punch list, not re-litigated in
the v9 score except where independently re-observed (e.g. dim 9).

## New-in-v9 material, reviewed

The v9 delta (k* falsification, per `changelog.md` and
`stale_claims_audit.md`) is cleanly executed: the stale-claims audit's
8 Part-A sites all read as claimed in the shipped `main.tex`, no group
median from the #476-affected display is quoted, and the supersession
of the v8 partial test (p=0.43) is stated inline. The audit's scope was
k*-only, which is precisely why the `recalibrated_verdict.json` drift
(a #456-era side effect) was not caught by it.
