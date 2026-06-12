---
title: "Bucket Brigade: A Parametric Cooperative-Competitive Benchmark with Computable Nash Equilibrium Structure"
authors:
  - "Robb Walters"
venue: "NeurIPS 2026 Cooperative AI Workshop"
target_length: "4 pages (workshop format), ~3000-4000 words excluding refs"
anonymous: false
tracker: rjwalters/bucket-brigade#357
issue: rjwalters/bucket-brigade#364
voice_notes: |
  Technical, MARL-literate audience that already knows Overcooked, Hanabi,
  SMAC, Melting Pot, MAgent, PettingZoo. Concise, honest, no marketing tone.
  The pitch is NE-transparency, NOT richness. Reviewers will reject anything
  that overclaims; over-disclaim where the trade is real (small state space,
  artificial geometry, single-repo origin).
prior_artifacts:
  - "paper/anvil_memo.env_spec.1/env_spec.md (formal env spec, §2)"
  - "paper/anvil_memo.ne_structure.1/ne_structure.md (analytical NE, §3)"
  - "paper/anvil_report.benchmark_comparison.1/report.md (related work, §5)"
---

# Brief: Bucket Brigade workshop paper

## One-sentence pitch

Bucket Brigade is a parametric cooperative-competitive MARL benchmark whose Nash equilibrium structure is computable per parameter cell, so the question "did the algorithm converge to the right equilibrium?" admits a ground-truth answer that none of the canonical MARL benchmarks (Overcooked, Hanabi, SMAC, Melting Pot, MAgent, PettingZoo) can provide.

## Why this paper

The MARL community evaluates new algorithms against benchmarks (Overcooked, Melting Pot, SMAC, …) whose equilibrium structure is not characterizable in closed form. As a result, when a new algorithm reports higher reward than PPO on Overcooked, the reader cannot tell whether the algorithm has converged to the "right" joint policy or merely to a Pareto-different one — there is no published NE to converge *to*. Bucket Brigade trades environment richness for equilibrium transparency: the state space is small enough to enumerate at minimal parameterizations, and the (β, κ, c) parameter family controls equilibrium structure continuously across cells (symmetric all-Work, asymmetric 1-Worker, no-pure-NE collapse). This makes Bucket Brigade a methodological complement to the canonical benchmarks — not a replacement — for the specific question of equilibrium-convergence quality.

The empirical hypothesis the paper will defend: **PPO success correlates with NE structure**. Symmetric-NE cells should converge reliably; asymmetric-only cells should cycle or settle into low-payoff mixed strategies (the "rest_trap" pattern); no-pure-NE cells should fail outright. Demonstrating this correlation, with the NE characterization treated as ground truth, is the core empirical contribution.

## Structure (4 pages target)

1. **Introduction** (~0.5pp) — Gap in MARL benchmarks: rich state spaces, opaque equilibrium structure. We invert.
2. **Environment** (~1pp) — Compressed from `anvil_memo.env_spec.1/env_spec.md` (247 lines → ~1 column). Game definition, parameter family, the three load-bearing scalars.
3. **Equilibrium structure** (~1pp) — Compressed from `anvil_memo.ne_structure.1/ne_structure.md` (294 lines → ~1 column). Mean-field single-house reduction → algebraic boundary inequalities → phase diagram (current 7-cell preview, gap-filled to 75 cells if available at submission). The central contribution.
4. **Trainability** (~0.75pp) — Empirical correlation between NE structure and PPO success. **Depends on #360 PPO sweep results**; placeholder until data arrives.
5. **Related work** (~0.5pp) — Compressed from `anvil_report.benchmark_comparison.1/report.md` (219 lines → ~half column). 6-row comparison table, honest positioning. Bucket Brigade complements, does not replace.
6. **Discussion / limitations** (~0.25pp) — Artificial, small, custom. Per Recommendation 3 of the benchmark_comparison report: defend the artificiality head-on.

References at the end; ~12-15 entries seeded in `refs.bib`.

## What is in scope

- Synthesis of the three completed Anvil artifacts (env_spec, ne_structure, benchmark_comparison) into a single 4-page workshop submission.
- One headline figure: NE phase diagram (PNG already at `experiments/nash/phase_diagram/phase_diagram.png`).
- One headline figure: PPO success rate per NE cell (PENDING #360 — placeholder figure for v1 draft).
- Honest reproducibility statement referencing `bucket_brigade` pip wheel (#373 — landed), `docs/PAPER_RESULTS.md` (#372 — landed), and HuggingFace baselines (#373 plumbing landed).

## What is out of scope

- Re-running any of the surveyed benchmarks (citation-only comparison, per `paper/_project.md`).
- Theoretical claims beyond the mean-field reduction. The analytical memo is honest about where the closed-form bounds disagree with empirics (κ-thresholds off by ~3-10×); the paper must inherit that honesty, not paper over it.
- "General-purpose MARL benchmark" framing. Per benchmark_comparison Recommendation 2: this paper says "complement, not replacement," consistently.
- Non-stationary payback matrix extension (deferred to a v2 paper per #366).

## Acceptance criteria (from #364)

- [ ] All cited results exist (no hand-waving "we believe X")
- [ ] At least one reviewer pass through Anvil critic loop with score ≥ 32/40 (advance) — paper rubric is /40, ≥35 is the publication-strength bar
- [ ] Reproducibility statement fully populated (pip-install one-liner, data, code, compute estimates)
- [ ] Submission-ready PDF compiles from `main.tex` + `refs.bib` via `pdflatex; bibtex; pdflatex; pdflatex`
- [ ] Page count: ≤4pp body excluding references (Cooperative AI Workshop format)

## Reference material in this thread

The drafter should treat `refs/` as the primary source of facts and quote from it liberally rather than reasoning fresh:

- `refs/env_spec.md` → §2 Environment (verbatim source for game definition, parameter table, reward decomposition)
- `refs/ne_structure.md` → §3 Equilibrium structure (verbatim source for the mean-field reduction, boundary inequalities, predicted vs. observed κ-thresholds)
- `refs/benchmark_comparison.md` → §5 Related work (verbatim source for the 6-row comparison table, the niche statement, and the limitations defense in §6)
- `refs/_project.md` → engagement context (voice norms, audience, what to avoid)

These files are kept inside the thread so the artifact is self-contained — the drafter does not need to reach outside `paper/anvil_pub.bb-workshop/` to produce a complete v1.

## Notes for the drafter (`pub-draft`)

- Use `\documentclass{anvil-paper}` for v1 (the workshop style file gets dropped in later under `templates/` per the venue-override pattern in `.anvil/skills/pub/SKILL.md` §Templates).
- Cite the analytical NE memo, env spec memo, and benchmark comparison report as in-repo references for now (`refs/ne_structure.md` etc.); the auditor will flag these for replacement with proper BibTeX before submission.
- Figure 1 (NE phase diagram) is rendered: copy from `experiments/nash/phase_diagram/phase_diagram.png` to `figures/phase_diagram.png` during draft.
- Figure 2 (PPO success heatmap) is **not yet rendered** — leave a `figures/ppo_heatmap.tex` TikZ placeholder with a TODO; the figurer fills it after #360 data lands.
- Keep §4 honest: if no PPO data exists at draft time, say so in the section ("PPO trainability data is pending and will land in v2 once #360 completes; v1 reports the analytical prediction only").

## Notes for the reviewer (`pub-review`)

- Apply the NeurIPS venue overlay (declared in `.anvil.json`). Generic /40 remains the gate.
- Hold §5 to the honest positioning standard set by `anvil_report.benchmark_comparison.1/`. The phrases "emergent cooperation," "general-purpose MARL benchmark," and "scalable to large populations" should not appear unqualified.
