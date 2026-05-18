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
  reward references across all 14 named scenarios + ``positional_default``,
  mirroring the value column of
  ``experiments/p3_specialization/diagnostics/random_baseline.py``'s
  ``SCENARIO_CITED_VALUES`` for non-torch consumers (issue #323).
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
#   MINSPEC_SPECIALIST = -22.07
#     Hand-coded ``SpecialistPolicy`` per-step team reward on the same
#     scenario. Re-derived post-#236 under issue #238 / PR #243; specialist
#     policies signal honestly (``signal == mode``) so the team-reward
#     distribution is unchanged from pre-#236 at n=50 precision.
#     Logs: ``experiments/p3_specialization/diagnostics/results/
#     issue238_post236_minspec/baselines.json``.
#
# Historical note (issue #293): prior to this constant being promoted to
# ``bucket_brigade.baselines``, three different MINSPEC_RANDOM values
# coexisted in the codebase — ``-96.07`` (pre-#246 2-dim sampler bug, PR
# #243), ``-92.92`` (n=50 corrected sampler, intermediate), and the canonical
# ``-87.72`` (n=1000 × 5 seeds, post-#246, PR #244). PR #250 fixed the sampler
# but didn't propagate the new value through all consumers; issue #293
# unified them on -87.72.
MINSPEC_RANDOM: float = -87.72
MINSPEC_SPECIALIST: float = -22.07

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
}

__all__ = [
    "SpecialistPolicy",
    "specialist_action",
    "specialist_action_joint",
    "MINSPEC_RANDOM",
    "MINSPEC_SPECIALIST",
    "SCENARIO_RANDOM_BASELINES",
]
