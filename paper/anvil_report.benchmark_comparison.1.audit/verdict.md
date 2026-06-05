# Audit verdict: anvil_report.benchmark_comparison v1

**Pass**: `pass: true`

**Findings count**: 8 total (0 blocker, 3 major, 5 minor)

## Critical flags

**None.** The report's OoM citation framing in Finding 5 was the candidate critical-flag site (it could have been "Unsupported quantitative claim"), but the report explicitly states the OoM framing IS the load-bearing claim, not precise integers — so it is uncited in the same sense that "approximately a thousand" is uncited but not unsupported. The reviser may verify and tighten in a future revision; not a blocker.

## Prior-report cross-check

`_project.md` lists `prior_reports: []` — this is the first artifact in the paper anvil track. No cross-engagement consistency checks required.

## Findings

See `findings.md` for the per-claim audit log and `evidence.md` for the citation traceability map.

## Audit summary

The report's load-bearing factual claims are:
1. Per-agent action sizes for Bucket Brigade (verified against in-repo source, Finding 1) — PASS.
2. State-space cardinality calculations (verified arithmetic, Finding 2) — PASS.
3. Per-benchmark action-space sizes (Overcooked=6, Melting Pot=8, Hanabi up-to-20, SMAC=6+n, MAgent=21, MPE=3–5) — PASS modulo the configurability caveats Appendix A acknowledges.
4. Year of canonical citation — PASS.
5. NE-structure characterization (Bucket Brigade tractable; others not) — PASS modulo the caveat that "tractable" for Bucket Brigade is a roadmap item (#358, #359), not a completed result. The report acknowledges this in Finding 7 and the Risks section.
6. Hetero-DO sweep characterization of v2_minimal as Hero-NE and rest-trap as 3FR+1FF — PASS (cited #355).
7. Citation counts as OoM — PASS-with-caveat (the OoM framing is itself the claim; precise counts not asserted).

Three "major" findings below are areas where a stricter audit standard would request tightening; the reviser may absorb them or defer to a future revision.

## Top revision priorities (pass: true; deferrable)

1. **Verify Hanabi info-set count** (Finding 2, "~10^11 reachable info-sets at 4P"): the Bard et al. 2020 paper discusses combinatorial scale qualitatively but the precise 10^11 number is a rough estimate from the paper's combinatorial argument. If the paper §6 reuses this number, double-check by computing or sourcing more precisely.
2. **Verify SMAC unit-feature dimensionality** (Finding 2): the Samvelyan et al. paper specifies per-unit feature vectors but the "tens of features per unit" claim is order-of-magnitude. A stricter audit would want the precise feature-vector length per agent (typically ~30 features per unit observed × unit-count).
3. **MAgent population scale** (Finding 2, Comparison Table): "10^3 agents on grids of side ~10^2" is consistent with the Zheng et al. paper's claims. A stricter audit would cite the specific configuration from the paper.
