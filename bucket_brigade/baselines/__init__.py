"""Reusable hand-coded baseline policies for the Bucket Brigade environment.

These baselines are deliberately scenario-agnostic — they take an observation
dict and an agent id and return a ``MultiDiscrete([num_houses, 2])`` action.
They are intended as reference policies for sanity-checking learned policies
(e.g., "can PPO beat a hand-coded specialist on scenario X?") and are kept in
the core ``bucket_brigade`` package rather than under ``experiments/`` so they
can be imported anywhere.

Currently exports:

- :func:`specialist_action` -- per-agent specialist that only fights fires on
  houses it owns. See :mod:`bucket_brigade.baselines.specialist`.
- :data:`MINSPEC_RANDOM`, :data:`MINSPEC_SPECIALIST` -- canonical per-step team
  reward references for the ``minimal_specialization`` scenario (see below).
- :data:`SCENARIO_RANDOM_BASELINES` -- canonical per-step uniform-random team
  reward references across all named scenarios (13 originals +
  ``positional_default`` + the two ``asym_*`` phase-diagram cells from #435),
  mirroring the value column of
  ``experiments/p3_specialization/diagnostics/random_baseline.py``'s
  ``SCENARIO_CITED_VALUES`` for non-torch consumers (issue #323).
- :data:`SCENARIO_GAP_REFERENCES` -- sparse per-scenario ``(random, upper
  reference)`` pairs for the Tier-1 ``gap_closed`` metric (issue #434). Only
  scenarios with hand-audited measurement provenance are listed; consumers
  must treat absent scenarios as *not scorable* rather than falling back to
  the MINSPEC constants (that silent fallback was the #434 bug). Degenerate
  entries (e.g. ``rest_trap``) may additionally carry trap-verdict anchors
  (``ne_per_step_bound``, ``scripted_best``) consumed by the four-way
  trap-escape verdict rule (issue #436,
  ``experiments/p3_specialization/run_tier1_cell.classify_trap_verdict``).

For the **frozen baseline distribution** (archetype pickles, Nash vectors,
PPO checkpoints shipped with the pip wheel + mirrored to HuggingFace), see
:mod:`bucket_brigade.baselines.release` (issue #373 / parent epic #365).

For the **reward-scale parity check** that packages
``SCENARIO_RANDOM_BASELINES`` as an executable check for downstream
consumers (``python -m bucket_brigade.baselines.parity``), see
:mod:`bucket_brigade.baselines.parity` (issue #437) and ``docs/PARITY.md``.
"""

from bucket_brigade.baselines.specialist import (
    SpecialistPolicy,
    specialist_action,
    specialist_action_joint,
)

# ---------------------------------------------------------------------------
# Canonical per-step team-reward references for ``minimal_specialization``
# ---------------------------------------------------------------------------
#
# Single source of truth (issue #293). All analyzers under
# ``experiments/p3_specialization/`` should import these constants rather than
# hardcoding the values, so that future re-derivations propagate consistently.
#
# Derivation provenance:
#
#   MINSPEC_RANDOM = -87.72
#     Uniform-random per-step team reward on ``minimal_specialization``
#     measured with the post-#236 3-dim sampler ``MultiDiscrete([10, 2, 2])``
#     (signal as first-class action). n=1000 episodes (200 episodes × 5 seeds
#     42..46), commit ``dffe1060``, issue #237 / PR #244. Logs:
#     ``experiments/p3_specialization/diagnostics/results/issue237_postmerge/``.
#     This is also the value recorded in
#     ``experiments/p3_specialization/diagnostics/random_baseline.py``'s
#     ``SCENARIO_CITED_VALUES["minimal_specialization"]["random"]``.
#
#   MINSPEC_SPECIALIST = -28.38
#     Hand-coded ``SpecialistPolicy`` per-step team reward on the same
#     scenario. Re-derived under issue #416 from the n=10k per-cell
#     calibration sweep landed in issue #413 / PR #415; n=10000 episodes
#     at the canonical cell (β=0.25, κ=0.50, c=0.50), seed=0,
#     ``bucket_brigade.baselines.per_cell.measure_specialist_homogeneous``.
#     Bootstrap 95% CI = [-29.748, -26.987].
#     Source: ``experiments/nash/phase_diagram/per_cell_baselines.json``
#     (first cell entry, ``cell_tag: canonical_b0.25_k0.50_c0.50``,
#     measured at commit ``12a9ba6a`` on alc-9). The same value was
#     reproduced under issue #416 at commit ``a476dc4e`` on alc-9 (seed=0,
#     n=10000) for ground-truth verification.
#
#     Historical n=50 value: ``-22.0717`` (issue #238 / PR #243), measured
#     with std 70.18 at n=50 from ``experiments/p3_specialization/
#     diagnostics/results/issue238_post236_minspec/baselines.json``. The
#     old mean was within ~0.64 SE of the new tight mean (delta 6.31 vs
#     SE 9.93) but outside the new n=10k 95% CI. The shift is a sampling-
#     precision correction, not a sampler-bug correction (cf. the #246
#     ``MINSPEC_RANDOM`` history below).
#
# Historical note (issue #293): prior to this constant being promoted to
# ``bucket_brigade.baselines``, three different MINSPEC_RANDOM values
# coexisted in the codebase — ``-96.07`` (pre-#246 2-dim sampler bug, PR
# #243), ``-92.92`` (n=50 corrected sampler, intermediate), and the canonical
# ``-87.72`` (n=1000 × 5 seeds, post-#246, PR #244). PR #250 fixed the sampler
# but didn't propagate the new value through all consumers; issue #293
# unified them on -87.72. The n=10k recalibration in PR #415 produced
# ``-87.93`` for ``random_homogeneous`` at the canonical cell (delta 0.21,
# inside even the tight CI); ``MINSPEC_RANDOM`` is intentionally left at
# the n=1000 × 5-seeds value because the shift is below measurement noise.
MINSPEC_RANDOM: float = -87.72
MINSPEC_SPECIALIST: float = -28.38

# ---------------------------------------------------------------------------
# Canonical per-step uniform-random team-reward references (all scenarios)
# ---------------------------------------------------------------------------
#
# Value-only mirror of
# ``experiments/p3_specialization/diagnostics/random_baseline.py``'s
# ``SCENARIO_CITED_VALUES[scenario]["random"]`` column. The full table in
# ``random_baseline.py`` also carries measurement metadata
# (``mlp_iter0``, ``note``); this mirror lifts only the per-step random
# baselines so non-torch consumers (e.g. ``experiments/scripts/
# compute_nash_trained.py``) can ``from bucket_brigade.baselines import
# SCENARIO_RANDOM_BASELINES`` without dragging in the training stack
# (``random_baseline.py`` transitively imports ``JointPPOTrainer`` ->
# ``torch``). See issue #323 / single-source-of-truth rationale.
#
# Derivation provenance: post-#236 (signal-as-first-class-action)
# re-derivation from issue #237 on ``COMPUTE_HOST_PRIMARY`` at commit
# ``dffe1060``: n=1000 episodes per scenario (200 episodes × 5 seeds
# 42..46), ``MultiDiscrete([10, 2, 2])`` uniform sampling. Logs committed
# under ``experiments/p3_specialization/diagnostics/results/
# issue237_postmerge/``. The ``minimal_specialization`` value here is the
# same canonical -87.72 published in ``MINSPEC_RANDOM`` above.
#
# ``asym_*`` provenance (issue #435): the two asymmetric_only NE
# phase-diagram cells promoted to named scenarios by #435 were measured
# with the same #237 protocol (n=1000 episodes = 200 episodes x 5 seeds
# 42..46, ``MultiDiscrete([10, 2, 2])`` uniform sampling, per-step =
# episode team reward / nights) via ``experiments/p3_specialization/
# diagnostics/random_baseline.py`` on host studio at commit ``866f43dd``.
# Both cells measure -78.27/step (95% bootstrap CI [-83.88, -72.81]) —
# **bit-identical**, not coincidentally close: in the bernoulli
# extinguish mode used by these scenarios the engine step order is
# extinguish -> burn_out -> spread -> spark, and burn_out ruins every
# still-burning house each night, so no house is ever BURNING when the
# spread phase runs and ``prob_fire_spreads_to_neighbor`` (beta) is inert.
# The committed phase diagram shows the same equivalence (identical NE
# payoffs across beta at fixed kappa in
# ``experiments/nash/phase_diagram/results.json``), so the two cells are
# a replication pair, not a sweep dimension.
#
# Drift guard: ``tests/test_baselines_constants.py`` asserts that every
# entry here matches ``SCENARIO_CITED_VALUES[scenario]["random"]`` so the
# two cannot silently diverge.
SCENARIO_RANDOM_BASELINES: dict[str, float] = {
    "default": 251.23,
    "easy": 355.07,
    "hard": 124.66,
    "trivial_cooperation": 399.99,
    "early_containment": 297.24,
    "greedy_neighbor": 292.78,
    "sparse_heroics": 246.06,
    "rest_trap": 302.87,
    "chain_reaction": 227.39,
    "deceptive_calm": 78.55,
    "overcrowding": 120.24,
    "mixed_motivation": 224.06,
    "minimal_specialization": -87.72,
    "positional_default": 250.73,
    "asym_b05_k09_c05": -78.27,
    "asym_b09_k09_c05": -78.27,
}

# ---------------------------------------------------------------------------
# Per-scenario gap_closed reference pairs (Tier-1 verdict ladder, issue #434)
# ---------------------------------------------------------------------------
#
# ``gap_closed = (trained - random) / (reference - random)`` is only
# meaningful when a scenario has BOTH a measured uniform-random baseline and
# a measured *upper* reference that beats random. This table is deliberately
# **sparse**: an entry exists only when the pair has hand-audited measurement
# provenance. Consumers (``experiments/p3_specialization/run_tier1_cell.py``)
# must resolve scenarios in three ways:
#
# 1. Entry with ``reference > random``  -> valid pair, fraction ladder applies.
# 2. Entry with ``reference = None`` (or ``reference <= random``) -> the
#    reference is *degenerate*: the fraction ladder MUST NOT be applied.
#    Report the scenario-scale ``uplift_over_random`` instead.
# 3. No entry -> not scorable. Do NOT fall back to the MINSPEC constants:
#    named scenarios differ by ~400 per-step reward units, so a MINSPEC-scale
#    fraction is never meaningful off ``minimal_specialization`` (that silent
#    fallback produced the vacuous ``gap_closed ~ 6.6`` on rest_trap, #434).
#
# Entry provenance:
#
#   minimal_specialization:
#     ``(MINSPEC_RANDOM, MINSPEC_SPECIALIST)`` — see the derivation comments
#     above. ``reference_kind = "specialist_homogeneous"`` (hand-coded
#     ``SpecialistPolicy``, n=10000, issue #416 / PR #415).
#
#   rest_trap:
#     ``random = SCENARIO_RANDOM_BASELINES["rest_trap"] = 302.87``/step
#     (issue #237 derivation, n=1000 episodes). Upper reference: **None**
#     (``degenerate_reason = "social_trap_ne_below_random"``). The frozen
#     Nash equilibrium (3×free_rider + 1×firefighter,
#     ``bucket_brigade/baselines/release/local/nash/rest_trap-v1.json``)
#     has ``team_payoff = 2984.04`` **per episode** — a per-EPISODE quantity,
#     while gap metrics are per-STEP. Do not divide by a guessed night count;
#     the bound "<= 248.7/step at >= 12 nights" (``ne_per_step_bound``,
#     min_nights = 12 for rest_trap) is sufficient to establish that the NE
#     sits *below* the 302.87/step random baseline. rest_trap's equilibrium
#     is team-suboptimal by construction (a social trap), so a
#     ``(trained - random)/(NE - random)`` fraction has a negative/degenerate
#     denominator. The entry therefore stays on the degenerate-reference
#     path (``reference = None``; the fraction ladder never applies), and
#     degenerate rows are instead classified by the four-way trap-escape
#     verdict rule (issue #436, ``run_tier1_cell.classify_trap_verdict``)
#     against three anchors: ``ne_per_step_bound``, ``random``, and the
#     measured ``scripted_best`` below.
#
#     ``scripted_best`` (issue #436 Part A): best all-scripted team profile
#     from ``experiments/p3_specialization/scripted_battery.py`` — the
#     homogeneous ``specialist`` (owned-house firefighter ×4) at
#     ``386.60/step`` (95% CI [386.17, 387.03], n=10000 episodes, seed=0,
#     host studio, commit ee21e796; paired Δ vs uniform = +83.67/step
#     [82.36, 84.89]). Committed artifact:
#     ``experiments/p3_specialization/scripted_battery/rest_trap.{json,md}``.
#     Drift guard: ``tests/test_baselines_constants.py`` asserts this block
#     matches the committed measurement artifact.
#
#     ``random_ci95_hi`` (issue #436 / PR #440 review): 95% CI upper bound
#     of the uniform-random anchor's own measurement — the battery's
#     final-stage n=10000 re-measurement of uniform play gives
#     ``302.94/step [301.46, 304.31]`` (same run/commit as
#     ``scripted_best``; stored in ``final.uniform.team.ci95_hi`` of the
#     committed artifact). The trap-escape rule's ``escaped_trap`` rung
#     requires the trained CI lower bound to clear THIS upper bound, not
#     the bare ``random`` point: the point anchor carries ±1.4/step
#     measurement noise at n=10k, so a sub-noise clearance of the point is
#     not a statistically supportable "above random" claim. This makes
#     rung 2 symmetric with rung 1 (which anchors on
#     ``scripted_best.ci95_hi``) and with the battery's own
#     ``beats_random`` check (``scripted_battery.py``). Drift guard:
#     ``tests/test_baselines_constants.py`` asserts it matches the
#     committed battery artifact.
#
# Drift guard: ``tests/test_baselines_constants.py`` asserts the ``random``
# side of every entry matches ``SCENARIO_RANDOM_BASELINES`` and that the
# minimal_specialization entry matches ``MINSPEC_RANDOM``/``MINSPEC_SPECIALIST``.
SCENARIO_GAP_REFERENCES: dict[str, dict[str, object]] = {
    "minimal_specialization": {
        "random": MINSPEC_RANDOM,
        "reference": MINSPEC_SPECIALIST,
        "reference_kind": "specialist_homogeneous",
        "provenance": (
            "MINSPEC_RANDOM (issue #237 / PR #244, n=1000x5 seeds) and "
            "MINSPEC_SPECIALIST (issue #416 re-derivation, n=10000 at the "
            "canonical cell b=0.25 k=0.50 c=0.50)."
        ),
    },
    "rest_trap": {
        "random": SCENARIO_RANDOM_BASELINES["rest_trap"],
        # 95% CI upper bound of the uniform-random anchor's own measurement
        # (issue #436 / PR #440 review): battery final-stage n=10000 uniform
        # re-measurement, 302.94/step [301.46, 304.31], seed=0 (stage seed
        # 500009), host studio, commit ee21e796. Artifact:
        # experiments/p3_specialization/scripted_battery/rest_trap.json
        # (final.uniform.team.ci95_hi). The escaped_trap rung of the trap
        # verdict requires the trained CI lower bound to clear this, not
        # the bare point above.
        "random_ci95_hi": 304.3071270072002,
        "reference": None,
        "reference_kind": None,
        "degenerate_reason": "social_trap_ne_below_random",
        # Upper bound on the frozen NE's per-step team payoff:
        # team_payoff (per episode) / min_nights. Conservative in the right
        # direction for the trap verdict ("significantly above the NE").
        "ne_per_step_bound": 2984.043694076538 / 12.0,
        # Measured all-scripted team upper anchor (issue #436 Part A). Used
        # by the trap-escape verdict rule only — deliberately NOT promoted
        # to ``reference``, so the gap_closed fraction ladder stays off for
        # this social-trap scenario.
        "scripted_best": {
            "value": 386.60293309775534,
            "ci95_lo": 386.1731053453639,
            "ci95_hi": 387.0264070404707,
            "kind": "scripted_battery:specialist",
            "n_episodes": 10000,
            "provenance": (
                "experiments/p3_specialization/scripted_battery.py, "
                "specialist (owned-house firefighter) x4, n=10000 episodes, "
                "seed=0, host studio, commit ee21e796; paired delta vs "
                "uniform = +83.67/step [82.36, 84.89]. Artifact: "
                "experiments/p3_specialization/scripted_battery/"
                "rest_trap.json"
            ),
        },
        "provenance": (
            "Frozen NE (rest_trap-v1.json, 3xfree_rider + 1xfirefighter) has "
            "team_payoff = 2984.04 PER EPISODE (<= 248.7/step at >= 12 "
            "nights), below the 302.87/step random baseline: rest_trap's "
            "equilibrium is team-suboptimal by construction (social trap). "
            "The gap fraction ladder is not applicable; degenerate rows are "
            "classified by the four-way trap-escape verdict (#436) against "
            "ne_per_step_bound / random (via its measured 95% upper bound "
            "random_ci95_hi) / scripted_best, with uplift_over_random as "
            "the quantitative headline."
        ),
    },
    # asymmetric_only NE phase-diagram cells (issue #435). Both entries are
    # deliberately identical: beta is inert in bernoulli extinguish mode
    # (see the SCENARIO_RANDOM_BASELINES provenance comment above), so the
    # two cells are the same effective environment. ``reference`` stays
    # None because the only committed upper anchor — the #358 double-oracle
    # NE team payoff, 72.0095 — is a PER EPISODE quantity while gap metrics
    # are per-STEP: the realized episode length under NE play is not
    # committed, so no per-step reference can be pinned without a new
    # measurement (positive payoff + min_nights = 12 only bounds it at
    # <= 6.0/step). NOT a rest_trap-style social trap: the NE per-step
    # value (positive) sits far above the -78.27/step random baseline, so
    # the #436 trap anchors (whose rung ordering assumes NE below random)
    # do not apply here either. Consumers report uplift_over_random.
    "asym_b05_k09_c05": {
        "random": SCENARIO_RANDOM_BASELINES["asym_b05_k09_c05"],
        "reference": None,
        "reference_kind": None,
        "degenerate_reason": "ne_reference_per_episode_only",
        "provenance": (
            "Random: issue #435 measurement, n=1000 (200 episodes x 5 "
            "seeds 42..46, #237 protocol), host studio, commit 866f43dd, "
            "-78.27/step [95% CI -83.88, -72.81]. Upper anchor: the #358 "
            "double-oracle NE for this cell (1xhero + 3xfirefighter, "
            "experiments/nash/phase_diagram/results.json, "
            "b0.50_k0.90_c0.50) has team payoff = 72.0095 PER EPISODE "
            "(<= 6.0/step at >= 12 nights) — per-episode vs per-step "
            "units, so it cannot serve as the fraction-ladder reference. "
            "Identical to asym_b09_k09_c05 by construction (beta inert in "
            "bernoulli mode)."
        ),
    },
    "asym_b09_k09_c05": {
        "random": SCENARIO_RANDOM_BASELINES["asym_b09_k09_c05"],
        "reference": None,
        "reference_kind": None,
        "degenerate_reason": "ne_reference_per_episode_only",
        "provenance": (
            "Random: issue #435 measurement, n=1000 (200 episodes x 5 "
            "seeds 42..46, #237 protocol), host studio, commit 866f43dd, "
            "-78.27/step [95% CI -83.88, -72.81]. Upper anchor: the #358 "
            "double-oracle NE for this cell (1xhero + 3xfirefighter, "
            "experiments/nash/phase_diagram/results.json, "
            "b0.90_k0.90_c0.50) has team payoff = 72.0095 PER EPISODE "
            "(<= 6.0/step at >= 12 nights) — per-episode vs per-step "
            "units, so it cannot serve as the fraction-ladder reference. "
            "Identical to asym_b05_k09_c05 by construction (beta inert in "
            "bernoulli mode)."
        ),
    },
}

__all__ = [
    "SpecialistPolicy",
    "specialist_action",
    "specialist_action_joint",
    "MINSPEC_RANDOM",
    "MINSPEC_SPECIALIST",
    "SCENARIO_RANDOM_BASELINES",
    "SCENARIO_GAP_REFERENCES",
]
