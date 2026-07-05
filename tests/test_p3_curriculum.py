"""Tests for the P3 episode-length curriculum (issue #260).

Pins down three behaviors of the optional ``CellConfig.curriculum`` schedule
added in #260:

1. **Default-off regression**: with ``curriculum=[]`` (default), the
   scenario's native ``min_nights`` is never mutated and per-iteration
   ``min_nights_floor`` records the native floor. This is the bit-identity
   guarantee for pre-curriculum runs.
2. **Phase transition**: a 2-phase curriculum flips
   ``trainer.env.scenario.min_nights`` and ``min_nights_floor`` at the
   correct iteration boundary.
3. **CLI parsing**: the ``--curriculum`` argparse type rejects malformed
   input loudly (no silent fallback) and accepts the documented
   ``'iter:min_nights,...'`` format.

The training runs are tiny (3-4 iters, 64 rollout steps, 2 agents) so the
full pytest stays fast; this is a unit test, not a smoke benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")  # skip module when RL extras absent (issue #484)

from experiments.p3_specialization.train import (  # noqa: E402
    CellConfig,
    _curriculum_floor_for,
    _parse_curriculum_arg,
    _validate_curriculum,
    train_one_cell,
)


# --- Helper-level tests (fast, no training) ---------------------------------


def test_validate_curriculum_empty_passes_through():
    """Empty schedule normalizes to empty list (curriculum disabled)."""
    assert _validate_curriculum([]) == []


def test_validate_curriculum_sorts_by_start_iteration():
    """Phases are sorted by ``start_iteration`` regardless of input order."""
    result = _validate_curriculum([[10, 8], [0, 5], [20, 12]])
    assert result == [[0, 5], [10, 8], [20, 12]]


def test_validate_curriculum_rejects_duplicate_start():
    """Duplicate ``start_iteration`` is rejected (ambiguous resolution)."""
    with pytest.raises(ValueError, match="duplicate start_iteration"):
        _validate_curriculum([[0, 5], [0, 8]])


def test_validate_curriculum_rejects_negative_iter():
    with pytest.raises(ValueError, match="start_iteration must be >= 0"):
        _validate_curriculum([[-1, 5]])


def test_validate_curriculum_rejects_nonpositive_floor():
    with pytest.raises(ValueError, match="min_nights floor must be > 0"):
        _validate_curriculum([[0, 0]])
    with pytest.raises(ValueError, match="min_nights floor must be > 0"):
        _validate_curriculum([[0, -3]])


def test_validate_curriculum_rejects_bad_arity():
    with pytest.raises(ValueError, match="2-element"):
        _validate_curriculum([[0, 5, 8]])


def test_validate_curriculum_rejects_non_int():
    with pytest.raises(ValueError, match="must be ints"):
        _validate_curriculum([[0.5, 5]])  # type: ignore[list-item]


def test_curriculum_floor_for_empty_returns_default():
    """No curriculum -> always return the scenario's native floor."""
    assert _curriculum_floor_for([], iteration=0, default_floor=12) == 12
    assert _curriculum_floor_for([], iteration=999, default_floor=12) == 12


def test_curriculum_floor_for_before_first_phase_returns_default():
    """If ``iteration`` precedes phase[0].start, the native floor still applies."""
    curriculum = [[5, 5], [10, 8]]
    assert _curriculum_floor_for(curriculum, iteration=0, default_floor=12) == 12
    assert _curriculum_floor_for(curriculum, iteration=4, default_floor=12) == 12


def test_curriculum_floor_for_picks_latest_active_phase():
    """The floor in effect is the latest phase whose ``start <= iteration``."""
    curriculum = [[0, 5], [17, 8], [34, 12]]
    assert _curriculum_floor_for(curriculum, iteration=0, default_floor=99) == 5
    assert _curriculum_floor_for(curriculum, iteration=16, default_floor=99) == 5
    assert _curriculum_floor_for(curriculum, iteration=17, default_floor=99) == 8
    assert _curriculum_floor_for(curriculum, iteration=33, default_floor=99) == 8
    assert _curriculum_floor_for(curriculum, iteration=34, default_floor=99) == 12
    assert _curriculum_floor_for(curriculum, iteration=1000, default_floor=99) == 12


def test_parse_curriculum_arg_empty_returns_empty():
    assert _parse_curriculum_arg("") == []
    assert _parse_curriculum_arg("   ") == []


def test_parse_curriculum_arg_canonical_format():
    assert _parse_curriculum_arg("0:5,17:8,34:12") == [[0, 5], [17, 8], [34, 12]]


def test_parse_curriculum_arg_tolerates_whitespace():
    assert _parse_curriculum_arg(" 0 : 5 , 17:8 ") == [[0, 5], [17, 8]]


def test_parse_curriculum_arg_rejects_missing_colon():
    with pytest.raises(argparse.ArgumentTypeError, match="missing ':' separator"):
        _parse_curriculum_arg("bogus")


def test_parse_curriculum_arg_rejects_non_integer():
    with pytest.raises(argparse.ArgumentTypeError, match="non-integer"):
        _parse_curriculum_arg("0:five")


def test_parse_curriculum_arg_propagates_validation_error():
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate"):
        _parse_curriculum_arg("0:5,0:8")


# --- Training-loop integration tests (tiny runs) ----------------------------


def _tiny_cfg(curriculum=None, num_iterations=3) -> CellConfig:
    """Smallest cell that still exercises the training loop end-to-end.

    Uses ``minimal_specialization`` (per the issue spec) with the scenario's
    canonical 4 agents, a small 128-step rollout, and the requested
    ``num_iterations``. Curriculum is optional. The 4-agent ownership
    reward vectors are hardcoded in the scenario factory, so we must
    match ``num_agents=4`` (matching scenario fixture in the project).
    """
    return CellConfig(
        scenario="minimal_specialization",
        lambda_red=0.0,
        seed=0,
        num_iterations=num_iterations,
        rollout_steps=128,
        num_agents=4,
        hidden_size=16,
        minibatch_size=32,
        curriculum=list(curriculum or []),
    )


def _load_metrics(output_dir: Path) -> list[dict]:
    with (output_dir / "metrics.json").open() as f:
        return json.load(f)


def test_train_cell_default_no_curriculum_preserves_native_min_nights(tmp_path):
    """With ``curriculum=[]``, the scenario's native ``min_nights`` is never mutated.

    Regression guard for the bit-identity invariant: pre-curriculum runs must
    behave exactly as they did before #260.
    """
    cfg = _tiny_cfg()
    out = tmp_path / "default_off"
    train_one_cell(cfg, out)

    metrics = _load_metrics(out)
    assert len(metrics) == cfg.num_iterations
    # ``minimal_specialization`` ships with min_nights=12.
    for rec in metrics:
        assert rec["min_nights_floor"] == 12, (
            f"iter {rec['iteration']}: expected min_nights_floor=12 "
            f"(native), got {rec['min_nights_floor']}"
        )


def test_train_cell_curriculum_transitions_floor_at_phase_boundary(tmp_path):
    """A 2-phase curriculum ``[[0, 5], [2, 8]]`` flips the floor at iter 2."""
    cfg = _tiny_cfg(curriculum=[[0, 5], [2, 8]], num_iterations=4)
    out = tmp_path / "two_phase"
    train_one_cell(cfg, out)

    metrics = _load_metrics(out)
    assert len(metrics) == 4
    floors = [rec["min_nights_floor"] for rec in metrics]
    # Phase 1 (iters 0..1) uses floor 5; phase 2 (iters 2..3) uses floor 8.
    assert floors == [5, 5, 8, 8], (
        f"Expected curriculum floors [5, 5, 8, 8], got {floors}"
    )


def test_train_cell_curriculum_late_start_uses_native_first(tmp_path):
    """If the first phase starts at iter > 0, early iters keep the native floor."""
    cfg = _tiny_cfg(curriculum=[[2, 5]], num_iterations=4)
    out = tmp_path / "late_start"
    train_one_cell(cfg, out)

    floors = [rec["min_nights_floor"] for rec in _load_metrics(out)]
    # Iters 0..1 keep native (12), iters 2..3 drop to 5.
    assert floors == [12, 12, 5, 5], f"Expected [12, 12, 5, 5], got {floors}"
