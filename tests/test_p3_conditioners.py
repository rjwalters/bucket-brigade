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

from bucket_brigade.analysis.info_theory import is_degenerate_conditioner
from experiments.p3_specialization.train import (
    _ACTION_NO_PRIOR_SENTINEL,
    _masked_mean_cmi_action,
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
    # Issue #235: action is now ``[house, mode, signal]`` (length 3) and
    # the pack is ``a[:, 0] * 4 + a[:, 1] * 2 + a[:, 2]`` (range 0..39).
    # All test action streams below use the honest convention
    # ``signal == mode`` so the packed code is
    # ``house * 4 + mode * 2 + mode = house * 4 + mode * 3``.

    def test_lag1_shifts_and_sentinels_t0(self):
        # Agent 0 takes the packed-action pattern. With honest signaling
        # the pack ``a[:, 0] * 4 + a[:, 1] * 2 + a[:, 2]`` reduces to
        # ``house * 4 + mode * 3``:
        #   [2, 1, 1] -> 8 + 3 = 11
        #   [3, 1, 1] -> 12 + 3 = 15
        #   [5, 1, 1] -> 20 + 3 = 23
        #   [1, 1, 1] -> 4 + 3 = 7
        a0 = np.array([[2, 1, 1], [3, 1, 1], [5, 1, 1], [1, 1, 1]])
        rollout = _make_rollout({0: a0, 1: a0.copy()})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        expected = np.array([_ACTION_NO_PRIOR_SENTINEL, 11, 15, 23], dtype=np.int64)
        np.testing.assert_array_equal(codes, expected)

    def test_lag_k_sentinels_first_k_steps(self):
        a0 = np.array([[1, 0, 0], [2, 1, 1], [3, 0, 0], [4, 1, 1], [0, 0, 0]])
        # packed (honest): 4, 11, 12, 19, 0
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=3)
        # First 3 entries are sentinel; then prior-prior-prior actions.
        assert codes[0] == _ACTION_NO_PRIOR_SENTINEL
        assert codes[1] == _ACTION_NO_PRIOR_SENTINEL
        assert codes[2] == _ACTION_NO_PRIOR_SENTINEL
        assert codes[3] == 4
        assert codes[4] == 11

    def test_short_rollout_all_sentinel_when_lag_ge_T(self):
        a0 = np.array([[1, 0, 0], [2, 1, 1]])
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
        # The pack is ``a[:, 0] * 4 + a[:, 1] * 2 + a[:, 2]`` (issue #235).
        # [7, 1, 1] -> 28 + 2 + 1 = 31; [9, 0, 0] -> 36 + 0 + 0 = 36.
        a0 = np.array([[7, 1, 1], [9, 0, 0]])
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        # codes[1] is the prior packed action (31); codes[0] is sentinel.
        assert codes[1] == 31

    def test_alphabet_max_is_39(self):
        # Largest legal packed action with action_dims = [10, 2, 2] is
        # ``9 * 4 + 1 * 2 + 1 = 39`` (< _ACTION_NO_PRIOR_SENTINEL = 40).
        a0 = np.array([[9, 1, 1], [0, 0, 0]])
        rollout = _make_rollout({0: a0})
        codes = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        assert codes[1] == 39
        # Sentinel value sits strictly above the legal range.
        assert _ACTION_NO_PRIOR_SENTINEL == 40

    def test_per_agent_independence(self):
        # Each agent's codes are computed from that agent's own action stream;
        # passing different action_dims values should not cross-contaminate.
        # Honest signaling so packed = house * 4 + mode * 3:
        # a0: [1,0,0]=4, [2,0,0]=8, [3,0,0]=12
        # a1: [5,1,1]=23, [6,0,0]=24, [7,1,1]=31
        a0 = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        a1 = np.array([[5, 1, 1], [6, 0, 0], [7, 1, 1]])
        rollout = _make_rollout({0: a0, 1: a1})
        codes0 = _other_agent_action_codes(rollout, agent_j=0, lag=1)
        codes1 = _other_agent_action_codes(rollout, agent_j=1, lag=1)
        assert codes0[1] == 4 and codes0[2] == 8
        assert codes1[1] == 23 and codes1[2] == 24

    def test_rejects_lag_zero(self):
        a0 = np.array([[1, 0, 0], [2, 0, 0]])
        rollout = _make_rollout({0: a0})
        with pytest.raises(ValueError):
            _other_agent_action_codes(rollout, agent_j=0, lag=0)


class TestMaskedMeanCMIAction:
    """Aggregate-only masking of degenerate per-pair Option-3 CMIs.

    See the Amendment section of
    ``research_notebook/2026-05-15_p3_conditioner_decision.md`` for the
    motivating finding: at λ=0, agent 1's policy converges to fully
    deterministic on `default`, which mathematically zeros the pair (0, 1)
    Option-3 CMI even though the other 5 of 6 pairs remain valid. The
    aggregate masks the bad pair; per-pair raw CMIs are still emitted by
    ``_measure_information`` for drilldown (tested implicitly via the
    train.py code path).

    Tests construct synthetic per-pair CMI dicts that mirror what the live
    ``_measure_information`` loop produces for a 4-agent setup. Pair
    convention: ``i < j``; conditioning agent is ``j`` (the larger index).
    Six pairs total: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3) → conditioning
    agents [1, 2, 3, 2, 3, 3].
    """

    # Conditioning-agent indices in the canonical pair-iteration order for
    # a 4-agent rollout. Same order the per-pair loop in
    # ``_measure_information`` emits.
    CONDITIONING_AGENTS_4 = [1, 2, 3, 2, 3, 3]

    def test_one_degenerate_conditioner_excluded(self):
        # Exactly one conditioning agent (agent 1) is degenerate. Pair (0, 1)
        # is the only pair conditioned on agent 1 — it must be excluded from
        # the aggregate. The other five pairs survive.
        cmi_values = [0.5, 0.4, 0.3, 0.6, 0.7, 0.2]  # six pairs
        per_agent_degenerate = {1: True, 2: False, 3: False}
        mean_pair, n_valid = _masked_mean_cmi_action(
            cmi_values, self.CONDITIONING_AGENTS_4, per_agent_degenerate
        )
        # Excluded value is cmi_values[0] (pair (0,1) → conditioning agent 1).
        expected = float(np.mean([0.4, 0.3, 0.6, 0.7, 0.2]))
        assert n_valid == 5
        assert mean_pair == pytest.approx(expected)

    def test_all_degenerate_returns_nan_and_zero_pairs(self):
        # Every conditioning agent collapsed (e.g., end-of-training run where
        # all three agent policies are deterministic). Aggregate must be NaN
        # and ``n_valid_pairs`` must be 0 so downstream analyzers can skip
        # the cell rather than averaging a meaningless zero.
        cmi_values = [0.5, 0.4, 0.3, 0.6, 0.7, 0.2]
        per_agent_degenerate = {1: True, 2: True, 3: True}
        mean_pair, n_valid = _masked_mean_cmi_action(
            cmi_values, self.CONDITIONING_AGENTS_4, per_agent_degenerate
        )
        assert n_valid == 0
        assert np.isnan(mean_pair)

    def test_no_degenerate_returns_unmasked_mean(self):
        # Sanity baseline: when no conditioning agent is degenerate the
        # aggregate matches the plain mean over all six pairs.
        cmi_values = [0.5, 0.4, 0.3, 0.6, 0.7, 0.2]
        per_agent_degenerate = {1: False, 2: False, 3: False}
        mean_pair, n_valid = _masked_mean_cmi_action(
            cmi_values, self.CONDITIONING_AGENTS_4, per_agent_degenerate
        )
        assert n_valid == 6
        assert mean_pair == pytest.approx(float(np.mean(cmi_values)))

    def test_is_degenerate_conditioner_agrees_on_modal_fraction_1(self):
        # End-to-end sanity check that the degeneracy flag the masker
        # consumes is the same one ``is_degenerate_conditioner`` produces
        # from a fully-collapsed action stream (modal_fraction = 1.0).
        T = 1024
        # All actions identical → packed code constant; n_distinct = 1.
        codes = np.zeros(T, dtype=np.int64)
        is_deg, diag = is_degenerate_conditioner(codes)
        assert is_deg is True
        assert diag["modal_fraction"] == 1.0
        assert diag["n_distinct"] == 1
