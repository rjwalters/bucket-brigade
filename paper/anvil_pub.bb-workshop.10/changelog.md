# changelog: anvil_pub.bb-workshop.9 → .10

## Trigger

v10 is the `pub-revise` pass consuming **all** v9 critic siblings:

- `anvil_pub.bb-workshop.9.review/` — 32/44, `advance: false`, no
  critical flags (generic `anvil-pub-v2` rubric; advisory
  `anvil-pub-neurips-v1` overlay 11/16).
- `anvil_pub.bb-workshop.9.audit/` — 2 critical flags (CF-1 artifact
  supersession, CF-2 Table 1 caption/descriptor contradiction) + 6
  non-critical notes (NC-1…NC-6).
- `anvil_pub.bb-workshop.9.numeric/` — deterministic pre-check, 0
  findings (advisory; nothing to consume).

**CF-1 resolution strategy**: of the auditor's two options (pin the
superseded revisions vs. recompute at HEAD), v10 takes option (b) —
every §4 primary-sweep statistic is recomputed from the artifacts at
repository HEAD and Figure 2 is re-rendered by the preserved
`figures/src/recalibrated_heatmap.py` from the HEAD
`experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
(regenerated at commit `a5b8ccdc` with n=20 means blended onto the 8
buy-down cells). This restores the §6 "reproduce byte-for-byte /
re-derivable" claims to literal truth and removes the ambiguous
citation-by-path. The superseded all-n=4 values are retained only as
explicitly-pinned provenance (ρ=0.007 at revision `22b1fda6`; the
withdrawn robustness claim's 0.0005 margin).

## New §4 headline numbers (all recomputed from HEAD artifacts)

| Quantity | v9 (superseded revision) | v10 (HEAD) | HEAD artifact |
|---|---|---|---|
| Class means, `gap_closed_ne`: sym / mixed / asym | 0.180 / 0.107 / 0.059 | 0.142 / 0.096 / 0.060 | `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json` (recomputed per class) |
| `no_convergence` homogeneous-fallback mean | −0.049 | −0.024 | same |
| Homogeneous class means: sym / mixed / asym / collapse | 0.051 / 0.050 / 0.033 / −0.049 | 0.041 / 0.046 / 0.030 / −0.024 | same |
| Adjacent class-mean separations quoted in Results | 0.001 / 0.017 / 0.082 (homogeneous) | 0.045 / 0.036 (on `gap_closed_ne`, NE-bearing classes) | same |
| Per-class mean within-cell std | ≈0.21–0.44 | ≈0.20–0.42 | same |
| Entropy headline (31 cells, mean h_cond vs `gap_closed_ne`) | ρ=0.007 (p=0.97) | ρ=0.109 (p=0.56), with ρ=0.007 pinned to revision `22b1fda6` as the original all-n=4 computation | `experiments/nash/phase_diagram/entropy_vs_trainability.{json,md}` |
| Nominally significant entropy entry | spread vs homog, p=0.039 | min vs homog, p=0.038 | same |
| β-column example / max within-column range | −0.32/−0.11/−0.00; up to 0.36 | −0.32/+0.01/−0.00; up to 0.34 | same |
| 4×-budget contrast | −0.108 (from −0.049) | −0.108 (from −0.024) | `…/phase_diagram_ppo_longbudget/cell_*/cell_summary.json` + verdict JSON |
| Metric-robustness sentence | "same qualitative ordering … robust to the metric choice" | **withdrawn** — sym/mixed inverts at HEAD (0.041 < 0.046); surviving claim scoped to NE-bearing > no-convergence and symmetric > asymmetric on both metrics | verdict JSON (recomputed) |

Unchanged-at-HEAD numbers were re-verified and left alone (k\* join
statistics, buy-down CI ratios 2.00–3.64×, 0/8 flips, 0.372→0.043,
per-cell std 0.99 / median 0.24, §5 anchor ladder, §3 Nash counts).

## Critic-note → change map

| Source | Note | Resolution |
|---|---|---|
| bb-workshop.9.audit (critical-flag CF-1) | §4/abstract class means, ρ=0.007 headline, β-column example, homogeneous-ordering claim, and Figure 2 all match superseded revisions of `recalibrated_verdict.json` / `entropy_vs_trainability.{json,md}`; §6 byte-for-byte claim false at HEAD | Recomputed every affected number from HEAD (table above); Figure 2 re-rendered from HEAD JSON via `figures/src/recalibrated_heatmap.py` (spot-check: κ=0.3, c=1.0, β=0.5 now +0.04, was +0.37); §4 protocol paragraph now declares the n=20/n=4 blend and cites the artifact path; "robust to the metric choice" withdrawn with the inversion stated explicitly; ρ=0.007 pinned to revision `22b1fda6` as provenance next to the HEAD ρ=0.109 headline (single headline, no dual-quoting); §6 byte-for-byte sentence now literally true and says so ("match the artifacts at repository HEAD") |
| bb-workshop.9.audit (critical-flag CF-2) | Table 1 caption contradicts table body (κ=0.3 row also totals 6) and "high-κ × c=0.5 corner" misdescribes the six unsampled cells (they are κ∈{0.3,0.7}, mid-grid) at three sites | All three sites fixed: §3 prose → "the six unsampled c=0.5 cells at κ∈{0.3,0.7}—the two mid-κ columns of that panel"; Table 1 caption → "row totals are 9 except at κ∈{0.3, 0.7} (6 cells each…)"; Figure 1 caption → "the κ∈{0.3,0.7} columns of the c=0.5 panel". No "corner"/"subsumed" language remains |
| bb-workshop.9.review (generic, major: overfull hboxes) | Five severe overfull hboxes 96.7–128.9pt (pp. 4, 10–12) from long `\texttt` artifact paths and unbreakable constructs | Added `xurl` + breakable `\path{}` for all long artifact paths (`\urldef` for the two caption-resident paths); §3 reward tuple moved to display math; §5 verdict ladder restructured as an itemized list; anchor-table provenance column converted to a ragged `p{}` column with paths moved to the caption. Final build log: **0 overfull hboxes of any size** (target was 0 > 10pt) |
| bb-workshop.9.review (generic, major: related work) + venue:neurips (major, novelty) | "Only published parametric MARL game" claim never positioned against the OpenSpiel / exploitability-evaluation (NashConv/PSRO) tradition | New §6 paragraph "The equilibrium-computation and exploitability-evaluation tradition" engaging OpenSpiel and PSRO/NashConv, crediting that line for measurable equilibrium convergence and scoping our claim to the *parametric-family / phase-diagram* property; abstract claim re-scoped ("to our knowledge the only published *parametric* MARL family … across a controlled grid"); `openspiel2019` + `psro2017` added to refs.bib (metadata from the canonical papers; flagged in the .bib comment for litsearch/audit verification like ppo2017/mappo2022). The reviewer's fuller recommendation (re-run `pub-litsearch`) remains open for the orchestrator |
| bb-workshop.9.review (generic, minor: abstract) | ~700-word abstract carrying per-test statistics; extract-in-90-seconds fails; also serves dim 9 | Abstract rewritten to ~318 rendered words; all per-test statistics (class means, ρ/p values, anchor ladder, CI bounds) moved to their owning sections; contribution-per-sentence structure kept |
| bb-workshop.9.review (generic, dim 9) | Conclusion restates the abstract at comparable length; caveats stated 3–4× | Conclusion tightened ~30% (statistics deduplicated, caveat restatements collapsed); abstract compression above removes the largest duplication. Appendix bulk itself retained per the operator's standing appendix policy (see nits) |
| bb-workshop.9.review (generic, minor: ordering-not-significance) | Percentile-bootstrap CIs on class means would make the ordering claim quantitative | **Declined** — no committed artifact carries class-mean bootstrap CIs, and this paper's standard is that every number traces to a committed artifact path; the Results paragraph already frames ordering-not-significance explicitly and the camera-ready 4×-cross-class sweep is named as the significance gate. Flagged for the camera-ready alongside that sweep |
| bb-workshop.9.review (v6 carry-forward) | App. A §A.6 named templates (Volunteer's Dilemma, Public Goods, Stag Hunt, free-rider) lack `\cite{}` links | **Declined (again)** — no verifiable source material on disk; adding textbook citations unverified would trade a polish item for citation-hygiene risk. Explicitly a `pub-litsearch` work item |
| bb-workshop.9.review (v6 carry-forward) | §B.4/§B.5 boundary reads as a layout accident | §B.4 given a real lead-in paragraph (why the 7-cell preview is retained) and a forward transition into §B.5 |
| bb-workshop.9.review (v6 carry-forward) | Small-q(k) Taylor linearisation not named among §B.5's "Five sources of systematic bias" | Added as a sixth named source ("Small-q(k) linearisation", with the T·q ≈ 0.24 worst-case corner quantified); §B.5 lead now reads "Six sources" |
| bb-workshop.9.review (v6 carry-forward) + venue:neurips (minor, reproducibility) | HuggingFace baselines pathway still "in progress" | **Declined** — the sentence is a true status statement; hosting is an operator task, not a prose fix. Venue checklist implication noted for submission time |
| bb-workshop.9.review (v6 carry-forward) | Figure 1 caption lacks the "three panels, one per c" half-sentence | Added ("Three panels, one per cost c∈{0.5,1.0,2.0}; within each panel rows are κ and columns are β") |
| bb-workshop.9.review (nit) | Figure 2 caption "top- and middle-right" imprecise for asymmetric cells at c=1.0 | Position gloss dropped; caption now says "asymmetric-only cells show positive but smaller gap closure" without a location claim |
| bb-workshop.9.review (nit) | 26 rendered pages vs. 4-page workshop body target | **Declined** — standing operator decision (appendix policy from the v6 cycle); v10 is 27 pages (new related-work paragraph + two footnotes, net of abstract/conclusion compression). Submission-time page-budget answer still owed |
| bb-workshop.9.audit (NC-1) | `ppo2017`, `mappo2022` unverified on disk | Kept — both are the canonically-correct references and the auditor's knowledge-based check found no metadata mismatch; refs.bib header still requests audit re-verification. Off-disk verification remains open (litsearch) |
| bb-workshop.9.audit (NC-2) | Six benchmark keys verify only against the author-supplied evidence index | No prose change (claims verified point-for-point against `refs/benchmark_comparison.md`); primary-PDF acquisition is a refs/ task, noted for the operator |
| bb-workshop.9.audit (NC-3) | `pettingzoo2021` / `mappo2022` appeared in the NeurIPS Datasets & Benchmarks track | Both entries now name the track in `booktitle` |
| bb-workshop.9.audit (NC-4) | "4×" denotes different multipliers in §4 (200×4096 = 8× env steps) vs §5 (200×2048 = true 4×) | Clarifying footnote added at the §4 usage, stating both conventions and that each label matches its committed artifact |
| bb-workshop.9.audit (NC-5) | 300/300 bootstrap-combination sweep and t-interval [304.67, 310.99] trace to the PR #460 judge recomputation, not a regenerable `experiments/` artifact | Provenance footnote added at the claim, naming the committed v8 paper-trail files and flagging the weaker-than-usual standard |
| bb-workshop.9.audit (NC-6) | No stale figures by mtime; Figure 2 reproducibility handled under CF-1 | No action needed (CF-1 resolution re-rendered Figure 2; `figures/src/` preserved verbatim) |
| bb-workshop.9.numeric | 0 findings over 1043 extracted numbers | No action needed |

## Build status

`pdflatex + bibtex + pdflatex ×2` against the shipped
`.anvil/skills/pub/templates/anvil-paper.cls`: exit 0, 27 pages,
**0 undefined references/citations**, **0 overfull hboxes** (v9: 10,
five severe at 96.7–128.9pt). `figures/src/` carried over verbatim;
`figures/recalibrated_heatmap.pdf` re-rendered from HEAD,
`figures/phase_diagram.pdf` unchanged.

## No-regression notes

- §§2–3 method/derivation content untouched except the display-math
  reward tuple and the CF-2 descriptor (dims 1–2 at 5/6 preserved).
- §4/§5 argumentation structure preserved; only the numbers, the
  withdrawn robustness sentence, and the blend disclosure changed.
- Citation hygiene: all 11 keys resolve; two new entries carry complete
  fields + arXiv IDs; no orphans.
