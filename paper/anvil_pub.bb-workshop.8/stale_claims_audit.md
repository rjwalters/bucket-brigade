# Stale-claim audit: anvil_pub.bb-workshop.7 → .8 (issue #462)

Scope mandated by issue #462: sweep v7 for every claim the three
post-v7 results change — (1) the 16× marginal trap escape (PR #460,
issue #444), (2) the rest_trap symmetric-DO cycling characterization
(PR #455, issue #445), (3) the 20-seed noise buy-down (PR #456, issue
#443) — and reconcile each against the committed artifacts
(`experiments/p3_specialization/tier1_runs_trap_escape/`,
`experiments/nash/rest_trap_seeded_do/RESULTS.md`,
`experiments/nash/phase_diagram/noise_buydown_precision.md`,
`experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`,
`docs/PAPER_RESULTS.md` §6–9) and the independent verification in the
judge review comments on PRs #455/#456/#460. The audit baseline is v7
(`paper/anvil_pub.bb-workshop.7/main.tex`), the latest revision in the
artifact trail.

Grep sweep of v7 `main.tex` (case-insensitive) for `indistinguishable`,
`at_random` / `at\_random`, `escaped`, `306.26` / `306.3`, `pending`,
`buy-down`, `may exist`, `interventional`, `rest_trap` / `rest\_trap`,
`equilibrium characteri`, and `n{=}4` contexts.

## Part A — result 1: 16× marginal statistical trap escape (PR #460)

v7 was written when the strongest trained result on `rest_trap-v1` was
standard-budget het_ppo at `at_random`. The 16× het_ppo `escaped_trap`
verdict (trailing-5 mean 307.83, seed-bootstrap 95% CI [305.00, 310.71],
lower bound clearing the 304.31 random-anchor upper bound by +0.69/step;
robust 300/300 bootstrap configurations; t-interval [304.67, 310.99] per
the PR #460 review) makes every unqualified "trained ≡ random" claim
stale. Because the flip is variance-driven (no dose-response on the
mean: 306.26 / 307.66 / 307.83, paired contrasts n.s.; uplift std
8.36 → 6.59) the corrections *qualify* rather than reverse: the hardness
headline stands, sharpened to "budget buys precision, not performance."

| # | Location (v7) | Old text (abridged) | Corrected text (v8, abridged) | Artifact citation |
|---|---|---|---|---|
| A1 | Abstract | "20-seed PPO reaches 306.3 — statistically indistinguishable from uniform-random play (302.9)" | "…20-seed PPO **at the standard budget** reaches 306.3 — statistically indistinguishable…"; new sentence: 4×/16× ladder, one marginal crossing (lo 305.00 vs 304.31), no dose-response, "budget buys seed-level precision, not performance," uplift ≈6% of headroom | `tier1_runs_trap_escape/16x/`; PAPER_RESULTS §9 |
| A2 | §1 contribution (4) | "the best 20-seed PPO run is statistically indistinguishable from random" | "20-seed PPO at the standard budget is statistically indistinguishable from random and a 16×-budget run separates from random only marginally and without mean improvement" | same |
| A3 | §5 headline italics | "the best trained policy is statistically indistinguishable from random play, while a hand-coded scripted team demonstrates ≈80/step of untouched headroom" | "at the standard training budget every trained policy is statistically indistinguishable from random play; at 16× budget exactly one configuration separates from random — marginally, and with no improvement in the mean — while a hand-coded scripted team demonstrates ≈80/step of headroom that no trained configuration approaches" | same |
| A4 | §5 Table 4 + caption | Four trained/scripted anchor rows; "Best trained (het_ppo, 20 seeds) 306.26 [302.95, 309.33]" | Row relabelled "het_ppo, standard budget"; new row "het_ppo, 16× budget (20 seeds) 307.83 [305.00, 310.71] — budget-scaling ladder; first escaped_trap"; caption explains the variance-driven flip | `tier1_runs_trap_escape/16x/` cell summaries |
| A5 | §5 at_random paragraph | "The strongest tier-1 result on rest_trap-v1 is heterogeneous PPO…" | "The strongest **standard-budget** tier-1 result…" (heading gains "at the standard budget") | `tier1_runs/tier1_verdict.md` (unchanged) |
| A6 | §5 asymmetry-rescue claim | "…therefore currently has *no* positive interventional evidence on this benchmark" | Claim kept, now cross-referenced: the 16× ladder is the sharpest interventional contrast to date (het_ppo crosses where ippo at identical budget does not, lo 303.46) but the trainers' means are statistically indistinguishable, so it is explicitly **not** counted as positive evidence | PAPER_RESULTS §9; `16x/tier1_verdict_notes.md` |
| A7 | §5 closing benchmark statement | Ladder "≤248.67 < 302.87 < 306.26 < 386.60 per step in which the next rung — escaped_trap — has a precise statistical meaning" (implicitly unreached) | Ladder "≤248.67 < 302.87 < 307.83 < 386.60"; "the escaped_trap rung has now been reached exactly once, marginally, by 16×-budget heterogeneous PPO"; remaining gap ≈79/step; "the open challenge is no longer statistical separation from random play; it is capturing headroom" | same |
| A8 | Conclusion | "the best 20-seed trained policy is statistically indistinguishable from random play" | "trained policies at the standard budget are statistically indistinguishable from random play, and a 16×-budget ladder yields exactly one marginal, variance-driven statistical escape with no improvement in the mean — budget buys precision, not performance" | same |

New §5 "Budget scaling" paragraph carries the full marginality apparatus
from PAPER_RESULTS §9: flat-mean paired contrasts (16×−1× = +1.57 ±
4.80/step, t = 1.46, n.s.), variance-driven CI-lower-bound march
(302.95 → 304.03 → 305.00), ≈6%-of-headroom uplift, best-seed ≈27%
(≈61/step short of `scripted_best`), CRN-coupled seed streams +
multiplicity caveat, pre-registered rule (#436/#440, stopping rule in
issue #444), and the explicit "never as 'PPO solves the trap at
scale'" sentence.

**Net Part A: 8 stale claim sites qualified/corrected; 0 claims
reversed** (the hardness headline survives, sharpened).

## Part B — result 2: rest_trap cycling characterization (PR #455)

| # | Location (v7) | Old text (abridged) | Corrected text (v8, abridged) | Artifact citation |
|---|---|---|---|---|
| B1 | §5 anchors caveat | "the frozen NE payoff carries unquantified double-oracle solver noise **and a better equilibrium may exist**" | "…carries unquantified double-oracle solver noise (a battery-seeded symmetric-oracle retry, below, searched for a better equilibrium and found nothing that stands)" | `rest_trap_seeded_do/RESULTS.md` §4 (anchor reconciliation: frozen NE unchanged) |

v7 contained no other claim about the rest_trap symmetric solve (the
missing-characterization hedge lived in that one sentence). The
characterization itself is **added** in v8 as a first-class paragraph
("Symmetric-oracle cycling is a property of the scenario, not a hole in
the characterization"): 50/50 iterations, min improvement 11.79 vs
ε = 0.01, payoff band [1611.56, 2316.61] (same basin as the unseeded
#352 run), mixture exploitable ≥407.50/episode (best deviation
`always_rest`), genome-mapped specialist team exploitable at every
position (+605 to +870/episode), cycling driven by 200-sim payoff noise
(support strategies score −258 to +237/episode at the verification
budget), asymmetric FR×3+FF NE (2984.04/episode) standing, and the
12/12 frozen-scenario coverage claim (11 converged + 1
exploitability-annotated non-converged entry). The RESULTS.md warning
that the log's "Cooperative behavior: 100.0%" line must not be quoted
as a finding is respected: v8 quotes no such figure. Abstract, §1
contribution (4), and Conclusion carry the closed-characterization
claim.

**Net Part B: 1 stale claim corrected; the positive 12/12 claim is an
addition enabled by the result.**

## Part C — result 3: 20-seed noise buy-down (PR #456)

| # | Location (v7) | Old text (abridged) | Corrected text (v8, abridged) | Artifact citation |
|---|---|---|---|---|
| C1 | Abstract | "the replacement coordination-threshold (k*) hypothesis is stated with its validation explicitly pending" | "a 20-seed noise buy-down shrinks CI half-widths 2.0–3.6× on an 8-cell subset while flipping none of the 8 trainability verdicts and re-verifying the entropy retirement, and the replacement k* hypothesis remains unvalidated (its class test is insignificant under a partial stratification)" | `noise_buydown_precision.md` |
| C2 | §1 contribution (3c) | "…k* hypothesis is stated with its validation explicitly pending" | buy-down results + "class test insignificant under a partial stratification; the full test is blocked on a pending per-cell k* artifact" | same |
| C3 | §4 k* paragraph | "the 20-seed noise buy-down on a k*-stratified cell subset needed to lift the n=4 noise ceiling **is an open issue** in the source repository (repo issue #443)" | The buy-down has run: partial-stratification class test (2 proven k*≥2 cells via thrust#259/#268 vs 6 k*=1 candidates) not significant (exact one-sided MW p = 0.43; exhaustive 28-assignment permutation p = 0.39), power caveat (min achievable p ≈ 0.036, "absence of evidence, not evidence of absence"); full test blocked on thrust#269 (open as of this revision) | `noise_buydown_precision.md` k* section |
| C4 | §4 noise-ceiling sentence | "at n=4 seeds the per-cell std … reaches 0.99 (median 0.24) — a noise ceiling that caps the achievable correlation for any predictor" | Sentence kept (historically true of the n=4 sweep); immediately followed by the new "noise ceiling, bought down at 20 seeds" paragraph: half-widths shrink 2.00–3.64× on `gap_closed_ne` (median 2.65; 1.84–3.59× homogeneous), consistent with √5 ≈ 2.24; 0/8 verdicts flip; entropy retirement re-verified (ρ = 0.109, p = 0.56); n=4 point estimates noise-dominated but directionally right (largest shift 0.372 → 0.043) | same; `recalibrated_verdict.json` (n=20 rows) |
| C5 | §4 stand-behind sentence | "the retirement of the entropy predictor and the β-invariance impossibility argument are the results we stand behind" | "…the retirement of the entropy predictor **(re-verified at n=20)** and the β-invariance impossibility argument…" | `entropy_vs_trainability.md` (refreshed) |
| C6 | Conclusion | "with the coordination-threshold k* hypothesis stated and its validation explicitly pending" | "re-verified by a 20-seed noise buy-down … flips none of the 8 re-measured verdicts, with the k* hypothesis still unvalidated (class test insignificant under a partial, low-power stratification; full test blocked on a pending per-cell k* artifact)" | `noise_buydown_precision.md` |
| C7 | §6 Threats | "per-class sample size is modest … the 4×-budget sweep rules out one budget-limited hypothesis … but not the cross-class question" | Appended: the 20-seed buy-down lifts the per-cell seed-noise ceiling on 8 of the 37 cells (verdicts unchanged) without yet covering the rest of the grid | same |

**Net Part C: 7 stale claim sites corrected** (all in the
"buy-down pending / n=4 noise ceiling" family; the negative result
itself is unchanged and strengthened).

## Part D — open threads added per issue #462 references

| Addition | Where | Traces to |
|---|---|---|
| ε-NE status of the phase-diagram asymmetric profiles under CRN re-evaluation: open | §3 (parenthetical after the solver-noise reading) and §6 Threats | repo issue #459 (filed during PR #455 review) |
| Superseded v1 recalibrated-table reproduction drift; v2 artifacts reproduce byte-for-byte | §6 Reproducibility | repo issue #461 (filed during PR #456 review) |

## Summary

- **Stale claim sites corrected: 16** (8 trained-vs-random family,
  Part A; 1 equilibrium-hedge, Part B; 7 buy-down-pending family,
  Part C). None of the corrections reverses a v7 conclusion: every one
  qualifies a claim whose unqualified form the new data outran.
- **Fresh claims added, each with a committed artifact path:** the 16×
  `escaped_trap` row + budget-scaling paragraph (variance-driven,
  marginal, headline sharpened); the rest_trap cycling
  characterization + 12/12 coverage; the n=20 precision table +
  re-verified entropy retirement + partial k* class test.
- **Marginality guard verified**: v8 contains no sentence readable as
  "PPO solves rest_trap" — the escape is described as "marginal,"
  "variance-driven," "no dose-response," "≈6% of headroom," and
  "never as 'PPO solves the trap at scale'" at every site where the
  verdict appears (abstract, §1, §5 ×3, conclusion).
- Figures, refs.bib, §2, Related work, and Appendices A/B: no claims
  affected by the three results; carried forward unchanged.
