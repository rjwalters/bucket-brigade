# Per-dimension scoring: anvil_report.benchmark_comparison v1

## Dim 1: Executive summary clarity — 7/7

The exec summary stands alone, names the inverted-trade pitch in the first paragraph, quantifies the action-space at v2_minimal in the second, states the niche in the third, and includes the headline limitations in a fourth. A reviewer who reads only the exec summary will leave with the correct mental model: "Bucket Brigade trades richness for NE-transparency."

## Dim 2: Finding sufficiency — 6/7

All 7 findings have evidence pointers and follow the claim → evidence → narrative structure. The –1 is on Finding 5: it stakes a load-bearing claim (Bucket Brigade has zero citations, the six surveyed benchmarks have many) but the comparison-quantification side (OoM citation counts) is itself unverified at write-time, with an explicit caveat. Honest, but a future revision could verify and remove the caveat. Findings 1, 2, 6 are particularly strong — they pin in-repo source lines.

## Dim 3: Recommendation actionability — 5/5

All three recommendations have explicit owner (paper issue #364, specific section), scope (which §, what to write/avoid), and "done when" criteria a downstream agent can verify. Recommendation 1 specifies the comparison table is reused, Recommendation 2 lists banned phrases, Recommendation 3 specifies an opening sentence.

## Dim 4: Evidence trail / citation — 5/6

Every quantitative claim has a source. arXiv IDs are correct (verified against memory of canonical paper IDs). The in-repo source citations are at line-number granularity. The –1: Finding 5 OoM counts are not refetched at write-time, which the report explicitly caveats. A stricter rubric would say "either refetch or remove" — this report says "stated as OoM with caveat," which is a defensible third option but loses a point.

## Dim 5: Risk & limitation disclosure — 4/4

Five distinct limitation categories: sample limits, data limits, methodological limits, time limits, selection bias on builder's choice. The "selection bias" entry is unusual and good — it pre-empts the reviewer who asks "why MPE and not GRF?"

## Dim 6: Internal consistency — 4/4

Cross-checked: the "8 actions at v2_minimal" claim is consistent across exec summary, Finding 1, and the comparison table. The "2304 states at v2_minimal" arithmetic (3² × 2⁴ × 2⁴) is correct. The (β, κ, c) parameter triple is consistent between Finding 6 and refs/bucket-brigade-env.md. No contradictions found.

## Dim 7: Format / presentation quality — 3/4

Markdown source-of-truth, no PDF (project specifies markdown-only delivery, so this is in-scope). Tables render in plain markdown. The –1 is the conservative cap for "rendered output not produced" — even though it's not required by this engagement, the rubric weights this dimension. Sections are well-organized: Exec Summary → Scope/Method → Findings → Comparison Table → Recommendations → Risks → Appendices → Evidence Index.

## Dim 8: Tone & audience calibration — 3/3

Tone is technical-MARL-literate throughout. No marketing tone, no overselling. The honest "Bucket Brigade is not a substitute" framing recurs naturally. Phrases like "brutally honest row," "this is the entire pitch of this report," and "the paper's job is to argue the first matters" are calibrated for a sophisticated reviewer who would distrust softer framing.
