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

__all__ = [
    "SpecialistPolicy",
    "specialist_action",
    "specialist_action_joint",
    "MINSPEC_RANDOM",
    "MINSPEC_SPECIALIST",
]
