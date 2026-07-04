# changelog: anvil_pub.bb-workshop.6 → .7

## Trigger

v7 is a **content revision** driven by the July-2026 session (repo issue
#446; PRs #431, #432, #433, #438, #439, #440, #441; issues #434, #436,
#442, #443), not a critic-loop revision. The v6 reviewer and auditor both
advanced v6 with 0 critical flags; v7 carries forward all v6 critic
conclusions and integrates four bodies of new committed evidence that the
paper's evidentiary base did not yet reflect. Every number added in v7
traces to a committed artifact path; the full stale-claim reconciliation
is committed alongside at `stale_claims_audit.md`.

## Headline changes

### 1. §4: per-cell trainability predictors — negative result + pending k* (PR #431, issue #443)

Three new paragraphs at the end of §4:

- **Conditional-entropy predictor retired.** Spearman ρ = 0.007
  (p = 0.97) for per-cell mean conditional entropy vs `gap_closed_ne`
  over the 31 converged cells; all four aggregates insignificant against
  both targets; the lone nominal hit (spread vs homogeneous, p = 0.039)
  dies under Bonferroni (α = 0.0063). Source:
  `experiments/nash/phase_diagram/entropy_vs_trainability.{py,json,md}`.
- **β-invariance impossibility argument.** Within each (κ, c) column the
  converged NE profile is identical across β (and the environment itself
  is bit-identical across β in bernoulli mode, per issue #442), while
  `gap_closed_ne` varies by up to 0.36 across β — so **no** statistic of
  the NE profile can explain within-column trainability variance.
- **k\* hypothesis stated with explicit pending status.** The
  coordination-threshold account is presented as a hypothesis, not a
  finding: the per-cell k\* artifact (rjwalters/thrust#269) has not
  landed and the 20-seed noise buy-down (repo issue #443) has not run
  (n = 4 per-cell std reaches 0.99, median 0.24 — a noise ceiling for
  any predictor). The k = 1 improvability-oracle datum
  (+13.4/step scripted single-agent headroom on the c = 0.5
  no-convergence cells, `improvability_oracle.md`, issue #428) is cited
  as a constraint on the account.

### 2. New §5: social-trap hardness (issues #434/#436; PRs #433/#438/#440)

New body section between the trainability sweep and Related work:

- **Four-anchor ladder table** (per-step team reward, `rest_trap-v1`):
  frozen NE ≤ 248.67 < always_rest 288.55 < uniform random 302.87
  (n=10k re-measurement 302.94 [301.46, 304.31]) < best trained
  (het_ppo, 20 seeds) 306.26 [302.95, 309.33] < scripted specialist
  386.60 [386.17, 387.03]. Sources:
  `experiments/p3_specialization/scripted_battery/rest_trap.{json,md}`,
  `experiments/p3_specialization/tier1_runs/tier1_verdict.md`,
  `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`.
- **Degenerate-reference mechanism + retraction of the vacuous pass.**
  The pre-#434 gap gate scored every scenario on
  minimal_specialization-scale references; on rest_trap uniform-random
  play alone mapped to gap_closed ≈ 6.58, so every historical "pass" was
  vacuous. The paper states this retraction explicitly.
- **het_ppo read as `at_random`, never a gate pass.** 20-seed CI lower
  bound clears the random *point* by 0.08/step but not the random
  anchor's measured n=10k upper bound (304.31); headline stays
  `uplift_over_random = +3.39 ± 7.34`/step. Stated consequence: the
  asymmetry-rescue hypothesis currently has **no** positive
  interventional evidence on this benchmark.
- **Trap-escape verdict rule** (#436) summarized: seed-bootstrap 95% CI
  lower bound walked up the NE / random-ci95-hi / scripted-best-ci95-hi
  ladder.

### 3. β-independence claims corrected throughout (issue #442)

Mechanistic finding: in bernoulli extinguish mode under the canonical
phase order, burn-out clears every burning house before spread runs, so
spread never fires and draws zero RNG — dynamics **and RNG stream** are
bit-identical across β. Consequences landed in v7:

- Abstract, §3 headline, and Conclusion no longer count β-independence
  as an empirical confirmation of the reduction (it holds by
  construction); the phase-order and collapse-regime predictions remain
  the confirmed structural claims.
- §3 adds the solver-noise reading: the splitting verdict row at
  (κ=0.9, c=0.5) and the cross-β payoff deltas (80.9 vs 72.0 there;
  −614.4 vs −648.0 at (κ=0.5, c=0.5); `results.json`) are measurements
  of double-oracle nondeterminism, and NE *payoffs* now carry an
  explicit unquantified-solver-noise hedge (also applied to the
  rest_trap NE payoff in §5, where the trap ordering is shown robust via
  the always_rest anchor).
- §4 re-reads cross-β PPO variation as a training-noise bound (feeds the
  §4 negative result).
- §6 Threats and Appendix B corrected: no ρ-sweep can activate β in
  bernoulli mode; falsifying the reduction's β treatment requires the
  contagion-active or continuous-extinguish variant.

### 4. §6: reproducibility subsection expanded (PRs #439/#441/#432)

- Frozen `-v1` scenario registry with bit-parity regression tests; paper
  numbers cite frozen IDs.
- Parity manifest + seconds-cheap CLI
  (`python -m bucket_brigade.baselines.parity --all`), CI-derived
  tolerance + scenario fingerprint (`docs/PARITY.md`).
- The verified ~7× downstream reward-scale mismatch (PR #432) as the
  motivating incident, with the reporting convention (frozen ID +
  manifest version + passing parity check).

### 5. §1 contributions updated

Added (3c) the negative result + β-invariance impossibility argument and
(4) the measured social-trap hardness result; honest-positioning bullet
renumbered to (5). Abstract extended to carry both new results.

## Not changed

Figures 1 and 2 and their source scripts are carried forward unchanged
from v6 (no new figure data). refs.bib unchanged (new material cites
in-repo artifact paths and repo issue/PR numbers, consistent with the
existing PAPER_RESULTS.md convention). §2 (environment), §Related work
table, and Appendices A/B structure unchanged except the two Appendix B
β paragraphs above.
