"""
Heterogeneous Double Oracle for asymmetric Nash equilibrium in 4-player games.

Finds asymmetric Nash equilibria via iterated best response from multiple
random starting profiles.  Unlike the symmetric Double Oracle (which assumes
all agents play the same strategy), this allows different positions to settle
on different strategies — the configuration required for role-differentiated
specialisation equilibria.

Algorithm
---------
For each random starting profile (θ_0, θ_1, θ_2, θ_3):
  1. For each position i in round-robin order, compute best response given the
     current strategies of the other three positions.
  2. If the BR improves payoff by > epsilon, update position i's strategy and
     add the BR to the shared pool.
  3. Repeat until no position improves in a full round (convergence) or
     max_iterations is reached.

Convergence ⟹ the profile is an epsilon-Nash equilibrium: no player can gain
more than epsilon by unilateral deviation.

Payoff evaluation uses core.run_heuristic_episode, which runs each episode
entirely in Rust and accepts a different strategy vector per agent.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize

import bucket_brigade_core as core
from bucket_brigade.envs.scenarios_generated import Scenario
from bucket_brigade.equilibrium.payoff_evaluator_rust import _convert_scenario_to_rust


# ---------------------------------------------------------------------------
# Worker function (module-level so it can be pickled by multiprocessing)
# ---------------------------------------------------------------------------


def _run_heterogeneous_sim(args: tuple) -> list[float]:
    """Run one episode with per-agent strategies, return all-agent rewards.

    Converts Python Scenario → Rust Scenario inside the worker so the Rust
    object is never pickled across process boundaries.
    """
    thetas, python_scenario, seed = args
    rust_scenario = _convert_scenario_to_rust(python_scenario)
    num_agents = python_scenario.num_agents
    agent_params = [np.asarray(t).tolist() for t in thetas]
    return core.run_heuristic_episode(rust_scenario, num_agents, agent_params, seed)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HeterogeneousNashEquilibrium:
    """
    Result from one iterated-best-response run.

    strategy_profile  – one 10-dim parameter vector per agent position
    payoffs           – expected payoff per position (MC estimate)
    team_payoff       – mean payoff across positions
    iterations        – rounds until convergence (or max_iterations)
    converged         – True if no position improved in the final round
    start_profile     – initial profile used for this restart
    """

    strategy_profile: list[np.ndarray]
    payoffs: list[float]
    team_payoff: float
    iterations: int
    converged: bool
    start_profile: list[np.ndarray] = field(repr=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HeterogeneousDoubleOracle:
    """
    Find asymmetric Nash equilibria by iterated best response from multiple starts.

    Parameters
    ----------
    scenario          : game scenario
    num_simulations   : MC episodes per payoff estimate
    opt_simulations   : MC episodes during gradient optimisation (cheaper)
    max_iterations    : rounds per restart before giving up
    epsilon           : minimum improvement (in raw payoff) to update a position
    seed              : master RNG seed
    num_restarts      : random starting profiles to try
    verbose           : print per-position progress
    num_workers       : parallel workers for MC evaluation
    """

    def __init__(
        self,
        scenario: Scenario,
        num_simulations: int = 1000,
        opt_simulations: int = 300,
        max_iterations: int = 25,
        epsilon: float = 2.0,
        seed: Optional[int] = None,
        num_restarts: int = 20,
        verbose: bool = True,
        num_workers: Optional[int] = None,
    ):
        self.scenario = scenario
        self.num_simulations = num_simulations
        self.opt_simulations = opt_simulations
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.seed = seed
        self.num_restarts = num_restarts
        self.verbose = verbose
        self.num_workers = (
            num_workers if num_workers is not None else min(cpu_count(), 8)
        )
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_team(
        self,
        strategy_profile: list[np.ndarray],
        n_sims: Optional[int] = None,
    ) -> list[float]:
        """MC estimate of per-position payoffs for the given profile."""
        n_sims = n_sims or self.num_simulations
        seeds = [self._rng.randint(0, 2**31 - 1) for _ in range(n_sims)]
        args_list = [(strategy_profile, self.scenario, s) for s in seeds]

        if self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(_run_heterogeneous_sim, args_list)
        else:
            results = [_run_heterogeneous_sim(a) for a in args_list]

        rewards = np.array(results)  # (n_sims, 4)
        return list(np.mean(rewards, axis=0))

    def _position_payoff(
        self,
        position: int,
        theta: np.ndarray,
        others: list[np.ndarray],
        n_sims: Optional[int] = None,
    ) -> float:
        """Payoff for `position` playing `theta` while others play `others`."""
        profile = list(others)
        profile.insert(position, theta)
        payoffs = self._evaluate_team(profile, n_sims=n_sims)
        return payoffs[position]

    # ------------------------------------------------------------------
    # Best-response computation for one position
    # ------------------------------------------------------------------

    def _best_response_for_position(
        self,
        position: int,
        profile: list[np.ndarray],
        strategy_pool: list[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        """
        Find an approximate best response for `position` given the other three
        positions' strategies.

        1. Score all strategies in the shared pool (cheap — already-evaluated).
        2. Continuous L-BFGS-B refinement from the best pool entry.
        3. Verify the candidate against the true (heterogeneous) evaluator.
        """
        others = [profile[j] for j in range(4) if j != position]

        # --- Step 1: pool scan ---
        best_pool_payoff = -np.inf
        best_pool_strategy = strategy_pool[0]
        for s in strategy_pool:
            p = self._position_payoff(position, s, others, n_sims=self.opt_simulations)
            if p > best_pool_payoff:
                best_pool_payoff = p
                best_pool_strategy = s.copy()

        # --- Step 2: L-BFGS-B refinement from best pool entry ---
        opt_seed = int(self._rng.randint(0, 2**31 - 1))
        # Build a cheap evaluator closure that uses opt_simulations
        _rng_local = np.random.RandomState(opt_seed)

        def _objective(theta: np.ndarray) -> float:
            # Sequential — called many times by L-BFGS-B; spawning a Pool
            # per call would dominate runtime.
            n = self.opt_simulations
            seeds = [_rng_local.randint(0, 2**31 - 1) for _ in range(n)]
            profile_t = [
                theta if j == position else others[j - (j > position)] for j in range(4)
            ]
            results = [
                _run_heterogeneous_sim((profile_t, self.scenario, s)) for s in seeds
            ]
            return -float(np.mean([r[position] for r in results]))

        res = minimize(
            _objective,
            x0=best_pool_strategy,
            bounds=[(0.0, 1.0)] * 10,
            method="L-BFGS-B",
            options={"maxiter": 50, "ftol": 1e-4},
        )
        opt_strategy = np.clip(res.x, 0.0, 1.0)

        # --- Step 3: verify with full simulations ---
        opt_payoff = self._position_payoff(
            position, opt_strategy, others, n_sims=self.num_simulations
        )

        if opt_payoff > best_pool_payoff:
            return opt_strategy, opt_payoff
        return best_pool_strategy, best_pool_payoff

    # ------------------------------------------------------------------
    # Single restart
    # ------------------------------------------------------------------

    def _run_restart(
        self,
        initial_profile: list[np.ndarray],
        strategy_pool: list[np.ndarray],
        restart_idx: int,
    ) -> HeterogeneousNashEquilibrium:
        profile = [s.copy() for s in initial_profile]

        for iteration in range(self.max_iterations):
            any_improved = False

            for position in range(4):
                others = [profile[j] for j in range(4) if j != position]
                current_payoff = self._position_payoff(
                    position, profile[position], others
                )
                br_strategy, br_payoff = self._best_response_for_position(
                    position, profile, strategy_pool
                )
                improvement = br_payoff - current_payoff

                if self.verbose:
                    print(
                        f"  restart={restart_idx:2d}  iter={iteration}  pos={position}  "
                        f"current={current_payoff:8.1f}  BR={br_payoff:8.1f}  "
                        f"Δ={improvement:+8.2f}",
                        flush=True,
                    )

                if improvement > self.epsilon:
                    profile[position] = br_strategy
                    # Add to shared pool if genuinely novel
                    if not any(
                        np.allclose(br_strategy, s, atol=1e-4) for s in strategy_pool
                    ):
                        strategy_pool.append(br_strategy.copy())
                    any_improved = True

            if not any_improved:
                payoffs = self._evaluate_team(profile)
                return HeterogeneousNashEquilibrium(
                    strategy_profile=profile,
                    payoffs=payoffs,
                    team_payoff=float(np.mean(payoffs)),
                    iterations=iteration + 1,
                    converged=True,
                    start_profile=initial_profile,
                )

        payoffs = self._evaluate_team(profile)
        return HeterogeneousNashEquilibrium(
            strategy_profile=profile,
            payoffs=payoffs,
            team_payoff=float(np.mean(payoffs)),
            iterations=self.max_iterations,
            converged=False,
            start_profile=initial_profile,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        initial_pool: Optional[list[np.ndarray]] = None,
    ) -> list[HeterogeneousNashEquilibrium]:
        """
        Find asymmetric Nash equilibria from multiple random starting profiles.

        Returns a list of HeterogeneousNashEquilibrium objects — one per
        restart.  Converged entries are epsilon-Nash equilibria; non-converged
        entries are the best profile found within max_iterations rounds.
        """
        from bucket_brigade.agents.archetypes import (
            FIREFIGHTER_PARAMS,
            FREE_RIDER_PARAMS,
            HERO_PARAMS,
            COORDINATOR_PARAMS,
            LIAR_PARAMS,
        )

        if initial_pool is None:
            strategy_pool = [
                FIREFIGHTER_PARAMS.copy(),
                FREE_RIDER_PARAMS.copy(),
                HERO_PARAMS.copy(),
                COORDINATOR_PARAMS.copy(),
                LIAR_PARAMS.copy(),
            ]
        else:
            strategy_pool = [s.copy() for s in initial_pool]

        archetype_names = ["FF", "FR", "Hero", "Coord", "Liar"]

        results: list[HeterogeneousNashEquilibrium] = []

        for restart_idx in range(self.num_restarts):
            # Sample 4 strategies from the current pool (may include BRs from
            # earlier restarts — pool grows during the run)
            n_pool = len(strategy_pool)
            indices = self._rng.randint(0, n_pool, size=4)
            initial_profile = [strategy_pool[i].copy() for i in indices]

            if self.verbose:
                names = [
                    archetype_names[i] if i < len(archetype_names) else f"BR{i}"
                    for i in indices
                ]
                print(
                    f"\n{'=' * 60}\n"
                    f"Restart {restart_idx + 1}/{self.num_restarts} — "
                    f"start profile: {names}",
                    flush=True,
                )

            eq = self._run_restart(initial_profile, strategy_pool, restart_idx)
            results.append(eq)

            if self.verbose:
                status = "CONVERGED" if eq.converged else "MAX_ITER"
                print(
                    f"  → {status} in {eq.iterations} rounds, "
                    f"team_payoff={eq.team_payoff:.1f}",
                    flush=True,
                )

        return results
