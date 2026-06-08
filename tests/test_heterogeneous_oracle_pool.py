"""
Tests for the persistent ``multiprocessing.Pool`` in ``HeterogeneousDoubleOracle``
(issue #389).

The phase-diagram driver (``compute_nash_phase_diagram.py``) was observed to
saturate a single core on 32-core cluster hosts, delivering ~1/32 of the
expected throughput. Root cause: ``_evaluate_team`` was constructing a fresh
``Pool`` per call, and the L-BFGS-B inner loop in
``_best_response_for_position`` ran its per-call ``opt_simulations`` MC sweep
entirely in the parent process to avoid the per-call Pool-construction cost.

The fix moves Pool lifetime to one ``Pool`` per ``solve()`` call and routes
every MC dispatch through ``_map_sims`` so the L-BFGS-B loop shares those
workers. These tests pin down the new invariants:

  - ``num_workers > 1`` ⇒ ``self._pool`` is a ``Pool`` instance while the
    solver is mid-``solve()`` and ``None`` outside it.
  - ``num_workers == 1`` ⇒ ``self._pool`` stays ``None`` throughout
    (preserves the sequential path the unit tests depend on).
  - ``_map_sims`` is the single dispatch point — every MC call funnels
    through it, including the L-BFGS-B inner loop. (Before the fix the
    L-BFGS-B inner loop bypassed any parallel mechanism by construction.)
  - Results computed with ``num_workers=1`` (sequential) and
    ``num_workers=2`` (Pool) match bit-for-bit for the same seed.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from bucket_brigade.envs import trivial_cooperation_scenario
from bucket_brigade.equilibrium.double_oracle_heterogeneous import (
    HeterogeneousDoubleOracle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solver(num_workers: int, seed: int = 13) -> HeterogeneousDoubleOracle:
    """Tiny-budget heterogeneous DO solver suitable for unit tests."""
    scenario = trivial_cooperation_scenario(num_agents=4)
    return HeterogeneousDoubleOracle(
        scenario=scenario,
        num_simulations=6,
        opt_simulations=3,
        max_iterations=1,
        epsilon=10.0,
        seed=seed,
        num_restarts=2,
        verbose=False,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoolLifecycle:
    """``self._pool`` invariants across ``solve()`` calls."""

    def test_pool_is_none_outside_solve(self):
        """The Pool only exists during a ``solve()`` call."""
        solver = _solver(num_workers=2)
        assert solver._pool is None
        solver.solve()
        assert solver._pool is None  # released after solve()

    def test_pool_active_inside_solve_when_parallel(self, monkeypatch):
        """With ``num_workers > 1`` the Pool is set before any restart runs."""
        solver = _solver(num_workers=2)
        original_run = solver._run_restart
        observed_pool_types: list[type | None] = []

        def spy_run_restart(initial_profile, strategy_pool, restart_idx):
            observed_pool_types.append(
                type(solver._pool) if solver._pool is not None else None
            )
            return original_run(initial_profile, strategy_pool, restart_idx)

        monkeypatch.setattr(solver, "_run_restart", spy_run_restart)
        solver.solve()

        # Every restart observed a live Pool. Without the fix
        # ``observed_pool_types`` would contain ``None`` entries because the
        # solver never set ``self._pool``.
        assert len(observed_pool_types) > 0
        for t in observed_pool_types:
            assert t is not None, (
                "self._pool should be set throughout solve() when num_workers > 1"
            )
            # ``mp.Pool`` is a factory (mp.pool.Pool is the class) — accept
            # either to avoid platform sensitivity.
            assert "Pool" in t.__name__

    def test_no_pool_when_sequential(self, monkeypatch):
        """``num_workers == 1`` must keep ``self._pool`` ``None``."""
        solver = _solver(num_workers=1)
        original_run = solver._run_restart
        observed = []

        def spy_run_restart(initial_profile, strategy_pool, restart_idx):
            observed.append(solver._pool)
            return original_run(initial_profile, strategy_pool, restart_idx)

        monkeypatch.setattr(solver, "_run_restart", spy_run_restart)
        solver.solve()

        assert len(observed) > 0
        assert all(p is None for p in observed), (
            "Sequential path (num_workers=1) must not allocate a Pool. "
            "Existing tests in test_heterogeneous_oracle_checkpoint.py rely "
            "on this for byte-identical behaviour vs the pre-#389 solver."
        )


class TestMapSimsDispatch:
    """All MC calls route through ``_map_sims`` — including the L-BFGS-B loop."""

    def test_map_sims_called_by_lbfgs_objective(self, monkeypatch):
        """
        L-BFGS-B in ``_best_response_for_position`` exercises ``_map_sims``.

        Regression for issue #389: before the fix the L-BFGS-B closure built
        its result list with a bare list comprehension over
        ``_run_heterogeneous_sim``, bypassing any pooled execution. The fix
        funnels every MC dispatch through ``_map_sims`` so the persistent
        Pool gets exercised on what is, by call count, the dominant per-cell
        workload.
        """
        solver = _solver(num_workers=1)  # sequential for deterministic test
        original = solver._map_sims
        call_count = {"n": 0}

        def counting(args_list):
            call_count["n"] += 1
            return original(args_list)

        monkeypatch.setattr(solver, "_map_sims", counting)
        solver.solve()

        # With 2 restarts × 1 max_iter × 4 positions, the BR loop alone
        # invokes ``_map_sims`` dozens of times. Tight lower bound here
        # is "anything > the pre-fix count", which was just the
        # ``_evaluate_team`` callers (~3-4 per BR). Asserting > 20 is a
        # generous floor that catches accidental regression to the
        # pre-fix list-comprehension path while staying robust to
        # L-BFGS-B convergence variability.
        assert call_count["n"] > 20, (
            f"_map_sims called only {call_count['n']} times — L-BFGS-B "
            "inner loop is likely bypassing the pooled dispatch."
        )


class TestSequentialParallelEquivalence:
    """Bit-identical equilibria across ``num_workers=1`` and ``num_workers>1``.

    This is the strongest correctness guarantee: the Pool change must be a
    pure parallelisation, not a semantic change. We seed both solvers
    identically, run both with ``progress_dir`` set (deterministic
    per-restart seeds, see ``solve``'s ``use_deterministic_per_restart``),
    and compare the resulting equilibrium signatures.
    """

    def test_results_match_across_worker_counts(self, tmp_path):
        # Pool spawning is slow on macOS (uses ``spawn`` start method).
        # Keep the budget small but representative.
        seq_solver = _solver(num_workers=1, seed=2026)
        par_solver = _solver(num_workers=2, seed=2026)

        seq = seq_solver.solve(progress_dir=tmp_path / "seq")
        par = par_solver.solve(progress_dir=tmp_path / "par")

        assert len(seq) == len(par)
        for a, b in zip(seq, par):
            assert a.converged == b.converged
            assert a.iterations == b.iterations
            assert a.team_payoff == pytest.approx(b.team_payoff, abs=1e-9)
            assert a.payoffs == pytest.approx(b.payoffs, abs=1e-9)
            for s_a, s_b in zip(a.strategy_profile, b.strategy_profile):
                assert list(map(float, s_a)) == pytest.approx(
                    list(map(float, s_b)), abs=1e-9
                )


def test_num_workers_default_is_cpu_count():
    """Default ``num_workers`` is ``cpu_count()`` — no historical ``min(_, 8)`` cap.

    Issue #389 documented that the prior cap left 24/32 cores idle on cluster
    hosts. The default is now ``cpu_count()``; callers cap explicitly via
    ``--num-workers`` if they want to share a host.
    """
    scenario = trivial_cooperation_scenario(num_agents=4)
    solver = HeterogeneousDoubleOracle(scenario=scenario, num_workers=None)
    assert solver.num_workers == mp.cpu_count()
