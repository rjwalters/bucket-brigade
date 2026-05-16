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
"""

from bucket_brigade.baselines.specialist import (
    SpecialistPolicy,
    specialist_action,
    specialist_action_joint,
)

__all__ = [
    "SpecialistPolicy",
    "specialist_action",
    "specialist_action_joint",
]
