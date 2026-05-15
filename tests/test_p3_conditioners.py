"""Tests for the P3 specialization CMI conditioner helpers.

Pins down ``_other_agent_action_codes`` (Option 3, added in #172) since it is
the per-pair sensitivity-check conditioner reported alongside the existing
Option 1 (``_state_summary_codes``) measurement in
``experiments/p3_specialization/train.py``. Architect rationale and the
four-way diagnostic table live in
``research_notebook/2026-05-15_p3_conditioner_decision.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest
import torch

from experiments.p3_specialization.train import (
    _ACTION_NO_PRIOR_SENTINEL,
    _other_agent_action_codes,
)


@dataclass
class _FakeRollout:
    """Minimal stand-in matching the fields ``_other_agent_action_codes`` reads.

    Only ``actions`` is consumed; using a dataclass keeps the test independent
    of the full ``RolloutBuffer`` definition (which carries observation /
    log-prob / value tensors irrelevant to the conditioner helper).
    """

    actions: Dict[int, torch.Tensor]


def _make_rollout(actions_np: Dict[int, np.ndarray]) -> _FakeRollout:
    return _FakeRollout(
        actions={i: torch.from_numpy(a.astype(np.int64)) for i, a in actions_np.items()}
    )


class TestOtherAgentActionCodes:
    def test_lag1_shifts_and_sentinels_t0(self):
        # Agent 0 takes the same packed-action pattern (5, 7, 11, 3); we
        # expect t=0 to be the sentinel and the rest to be the prior step.
        a0 = np.array([[2, 1], [3, 1], [5, 1], [1, 1]])  # packed = 5, 7, 11, 3
        rollout = _make_rollout({0: a0, 1: a0.copy()})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        expected = np.array([_ACTION_NO_PRIOR_SENTINEL, 5, 7, 11], dtype=np.int64)
        np.testing.assert_array_equal(codes, expected)

    def test_lag_k_sentinels_first_k_steps(self):
        a0 = np.array([[1, 0], [2, 1], [3, 0], [4, 1], [0, 0]])
        # packed = 2, 5, 6, 9, 0
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=3)
        # First 3 entries are sentinel; then prior-prior-prior actions.
        assert codes[0] == _ACTION_NO_PRIOR_SENTINEL
        assert codes[1] == _ACTION_NO_PRIOR_SENTINEL
        assert codes[2] == _ACTION_NO_PRIOR_SENTINEL
        assert codes[3] == 2
        assert codes[4] == 5

    def test_short_rollout_all_sentinel_when_lag_ge_T(self):
        a0 = np.array([[1, 0], [2, 1]])
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=2)
        # T == lag → entire output is sentinel (no t with a defined prior).
        np.testing.assert_array_equal(
            codes, np.full(2, _ACTION_NO_PRIOR_SENTINEL, dtype=np.int64)
        )
        # T < lag → likewise all sentinel.
        codes_3 = _other_agent_action_codes(rollout, agent_j=0, lag=3)
        np.testing.assert_array_equal(
            codes_3, np.full(2, _ACTION_NO_PRIOR_SENTINEL, dtype=np.int64)
        )

    def test_pack_matches_existing_convention(self):
        # The pack is ``a[:, 0] * 2 + a[:, 1]`` and must match the
        # convention used by the existing ``action_entropy`` metric in
        # ``_measure_information`` (lines 287-289 of train.py).
        a0 = np.array([[7, 1], [9, 0]])  # packed = 15, 18
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        # codes[1] is the prior packed action (15); codes[0] is sentinel.
        assert codes[1] == 15

    def test_alphabet_max_is_19(self):
        # Largest legal packed action with action_dims = [10, 2] is
        # ``9 * 2 + 1 = 19`` (< _ACTION_NO_PRIOR_SENTINEL = 20).
        a0 = np.array([[9, 1], [0, 0]])
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        assert codes[1] == 19
        # Sentinel value sits strictly above the legal range.
        assert _ACTION_NO_PRIOR_SENTINEL == 20

    def test_per_agent_independence(self):
        # Each agent's codes are computed from that agent's own action stream;
        # passing different action_dims values should not cross-contaminate.
        a0 = np.array([[1, 0], [2, 0], [3, 0]])  # packed = 2, 4, 6
        a1 = np.array([[5, 1], [6, 0], [7, 1]])  # packed = 11, 12, 15
        rollout = _make_rollout({0: a0, 1: a1})
        codes0 = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        codes1 = _other_agent_action_codes(rollout, agent_j=1, lag=1)
        assert codes0[1] == 2 and codes0[2] == 4
        assert codes1[1] == 11 and codes1[2] == 12

    def test_rejects_lag_zero(self):
        a0 = np.array([[1, 0], [2, 0]])
        rollout = _make_rollout({0: a0})
        with pytest.raises(ValueError):
            _other_agent_action_codes(rollout, agent_j=0, lag=0)
