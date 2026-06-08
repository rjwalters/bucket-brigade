"""
Tests for per-restart checkpointing in HeterogeneousDoubleOracle (issue #388).

The phase-diagram driver runs the asymmetric Nash solver in cells that take
many hours apiece; a host crash mid-cell used to lose every completed restart
inside that cell. ``HeterogeneousDoubleOracle.solve`` now writes
``restarts_progress.json`` after every successfully-completed restart and
replays already-completed restarts on resume, capping the worst-case loss at
one restart instead of one cell.

These tests use the smallest possible budget compatible with the solver
contract (no compute time on the heavy Nash workload — this is the
``trivial_cooperation`` scenario at ~10 simulations per call).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bucket_brigade.envs import trivial_cooperation_scenario
from bucket_brigade.equilibrium.double_oracle_heterogeneous import (
    PROGRESS_FILENAME,
    HeterogeneousDoubleOracle,
    HeterogeneousNashEquilibrium,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solver(seed: int = 123) -> HeterogeneousDoubleOracle:
    """Build a tiny-budget solver suitable for unit tests (~seconds, not hours).

    The verdicts produced here are not meaningful — we only care about
    checkpoint semantics: how many restarts ran, whether the file is on disk
    and parseable, and whether the resumed run is byte-equivalent to an
    uninterrupted one.
    """
    scenario = trivial_cooperation_scenario(num_agents=4)
    return HeterogeneousDoubleOracle(
        scenario=scenario,
        num_simulations=8,
        opt_simulations=4,
        max_iterations=1,
        epsilon=10.0,
        seed=seed,
        num_restarts=4,
        verbose=False,
        num_workers=1,
    )


def _eq_signature(eq: HeterogeneousNashEquilibrium) -> tuple:
    """Stable signature of an equilibrium result, used for equality checks."""
    return (
        bool(eq.converged),
        int(eq.iterations),
        tuple(round(float(p), 6) for p in eq.payoffs),
        round(float(eq.team_payoff), 6),
        tuple(tuple(round(float(v), 6) for v in t) for t in eq.strategy_profile),
        tuple(tuple(round(float(v), 6) for v in t) for t in eq.start_profile),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Per-restart checkpointing for HeterogeneousDoubleOracle.solve()."""

    def test_no_progress_dir_writes_nothing(self, tmp_path):
        """Default (no progress_dir) path is byte-identical to historical behaviour."""
        solver = _make_solver()
        results = solver.solve()
        assert len(results) == 4
        # Nothing should have been written anywhere.
        assert not (tmp_path / PROGRESS_FILENAME).exists()

    def test_progress_file_grows_per_restart(self, tmp_path):
        """``restarts_progress.json`` exists with N entries after N restarts."""
        solver = _make_solver()
        results = solver.solve(progress_dir=tmp_path)

        assert len(results) == 4
        progress_path = tmp_path / PROGRESS_FILENAME
        assert progress_path.exists()

        with open(progress_path) as f:
            payload = json.load(f)

        assert payload["schema_version"] == 1
        assert payload["num_restarts_target"] == 4
        assert payload["completed_restarts"] == [0, 1, 2, 3]
        assert len(payload["equilibria"]) == 4
        # Strategy pool grows at least to the seed-pool size (5 archetypes).
        assert len(payload["strategy_pool"]) >= 5
        for genome in payload["strategy_pool"]:
            assert len(genome) == 10
            assert all(isinstance(v, float) for v in genome)
        assert payload["best_so_far"] is not None
        assert "restart_idx" in payload["best_so_far"]
        assert "team_payoff" in payload["best_so_far"]

    def test_resume_after_simulated_crash(self, tmp_path, monkeypatch):
        """
        Crash after K of N restarts; resume; verify only the missing restarts
        run and the final result matches an uninterrupted run.
        """
        # ----- Phase 1: uninterrupted reference run -----
        ref_solver = _make_solver(seed=7)
        ref_results = ref_solver.solve(progress_dir=tmp_path / "ref")

        # Sanity: reference produced the expected number of restarts.
        assert len(ref_results) == 4

        # ----- Phase 2: interrupted run -----
        crash_dir = tmp_path / "crash"
        crash_solver = _make_solver(seed=7)

        original_run_restart = crash_solver._run_restart
        call_count = {"n": 0}
        K = 2  # let 2 of 4 restarts complete, then explode

        def flaky_run_restart(initial_profile, strategy_pool, restart_idx):
            if call_count["n"] >= K:
                raise RuntimeError("simulated host crash mid-restart")
            call_count["n"] += 1
            return original_run_restart(initial_profile, strategy_pool, restart_idx)

        monkeypatch.setattr(crash_solver, "_run_restart", flaky_run_restart)

        with pytest.raises(RuntimeError, match="simulated host crash"):
            crash_solver.solve(progress_dir=crash_dir)

        # Checkpoint must have K completed restarts on disk.
        progress_path = crash_dir / PROGRESS_FILENAME
        assert progress_path.exists()
        with open(progress_path) as f:
            mid_payload = json.load(f)
        assert mid_payload["completed_restarts"] == list(range(K))
        assert len(mid_payload["equilibria"]) == K

        # ----- Phase 3: resume on a fresh solver instance -----
        resume_solver = _make_solver(seed=7)
        # Record how many *new* restart calls happen post-resume.
        resume_calls = {"n": 0}
        original_resume_run_restart = resume_solver._run_restart

        def counting_run_restart(initial_profile, strategy_pool, restart_idx):
            resume_calls["n"] += 1
            return original_resume_run_restart(
                initial_profile, strategy_pool, restart_idx
            )

        monkeypatch.setattr(resume_solver, "_run_restart", counting_run_restart)

        resume_results = resume_solver.solve(progress_dir=crash_dir)

        # Only the missing restarts (N - K) should have actually run.
        assert resume_calls["n"] == 4 - K

        # All N restarts present in the final result list.
        assert len(resume_results) == 4

        # Final summary equals the uninterrupted reference run.
        ref_sigs = [_eq_signature(e) for e in ref_results]
        resume_sigs = [_eq_signature(e) for e in resume_results]
        assert ref_sigs == resume_sigs, (
            "Resumed run must produce the same equilibria as the uninterrupted "
            "reference run when both use the same progress_dir and seed."
        )

        # Final checkpoint contains all N entries.
        with open(progress_path) as f:
            final_payload = json.load(f)
        assert final_payload["completed_restarts"] == [0, 1, 2, 3]
        assert len(final_payload["equilibria"]) == 4

    def test_corrupt_checkpoint_starts_fresh(self, tmp_path):
        """A garbage / truncated checkpoint must not crash the solver."""
        progress_path = tmp_path / PROGRESS_FILENAME
        progress_path.write_text("{ this is not valid json")

        solver = _make_solver()
        results = solver.solve(progress_dir=tmp_path)
        assert len(results) == 4

        # The corrupt file was overwritten with a real checkpoint.
        with open(progress_path) as f:
            payload = json.load(f)
        assert payload["completed_restarts"] == [0, 1, 2, 3]

    def test_atomic_write_no_stray_tmp_files(self, tmp_path):
        """Successful run leaves no ``*.tmp`` siblings of the progress file."""
        solver = _make_solver()
        solver.solve(progress_dir=tmp_path)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"Unexpected temp files left behind: {leftovers}"

    def test_resume_when_all_complete_is_noop(self, tmp_path, monkeypatch):
        """If every restart is already in the checkpoint, no restart re-runs."""
        # Phase 1: complete a run.
        solver1 = _make_solver(seed=99)
        results1 = solver1.solve(progress_dir=tmp_path)
        assert len(results1) == 4

        # Phase 2: a fresh solver should run zero restarts.
        solver2 = _make_solver(seed=99)
        call_count = {"n": 0}
        original = solver2._run_restart

        def counting(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(solver2, "_run_restart", counting)
        results2 = solver2.solve(progress_dir=tmp_path)

        assert call_count["n"] == 0
        assert len(results2) == 4
        # The replayed equilibria match the originals.
        assert [_eq_signature(e) for e in results1] == [
            _eq_signature(e) for e in results2
        ]


class TestSerialisationHelpers:
    """Lightweight round-trip tests for the JSON (de)serialisation helpers."""

    def test_eq_state_round_trip(self):
        eq = HeterogeneousNashEquilibrium(
            strategy_profile=[np.linspace(0.1, 0.9, 10) for _ in range(4)],
            payoffs=[1.0, 2.0, 3.0, 4.0],
            team_payoff=2.5,
            iterations=7,
            converged=True,
            start_profile=[np.zeros(10) for _ in range(4)],
        )
        state = HeterogeneousDoubleOracle._eq_to_state(eq)
        # Round-trip through JSON to catch numpy-type leaks.
        state = json.loads(json.dumps(state))
        eq2 = HeterogeneousDoubleOracle._eq_from_state(state)

        assert eq2.converged is True
        assert eq2.iterations == 7
        assert eq2.team_payoff == pytest.approx(2.5)
        assert eq2.payoffs == pytest.approx([1.0, 2.0, 3.0, 4.0])
        for a, b in zip(eq.strategy_profile, eq2.strategy_profile):
            np.testing.assert_allclose(a, b)
        for a, b in zip(eq.start_profile, eq2.start_profile):
            np.testing.assert_allclose(a, b)
