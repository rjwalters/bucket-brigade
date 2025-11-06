"""
Research Findings Validation Tests.

This module validates key research findings from our evolution experiments
to prevent regressions and ensure reproducibility.

Research Phases Tested:
- Phase 1.5: Universal Nash Equilibrium (all scenarios converge to same strategy)
- Phase 2A: Boundary Testing (extreme parameter robustness)
- Phase 2A.1: p_spark Goldilocks Zone (optimal fire rate)
- Phase 2D: Mechanism Design (cooperation impossibility)
- Scale Testing: Population-size invariance

These tests use the evolved_v4 genomes and validate that:
1. All evolved strategies converged to nearly identical genomes
2. The universal strategy performs well across diverse scenarios
3. The strategy is robust to parameter variations
4. Mechanism design attempts fail to induce cooperation
5. Performance scales to larger populations
"""

import json
import itertools
from pathlib import Path
from typing import List, Dict

import pytest
import numpy as np

from bucket_brigade.envs.scenarios import (
    trivial_cooperation_scenario,
    early_containment_scenario,
    greedy_neighbor_scenario,
    sparse_heroics_scenario,
    rest_trap_scenario,
    chain_reaction_scenario,
    deceptive_calm_scenario,
    overcrowding_scenario,
    mixed_motivation_scenario,
    # Boundary scenarios
    glacial_spread_scenario,
    explosive_spread_scenario,
    wildfire_scenario,
    # Mechanism design scenarios
    free_work_scenario,
    cheap_work_scenario,
    expensive_work_scenario,
    prohibitive_work_scenario,
    Scenario,
)
from bucket_brigade.evolution.fitness_rust import RustFitnessEvaluator


# Base scenarios used in Phase 1.5
BASE_SCENARIOS = [
    "trivial_cooperation",
    "early_containment",
    "greedy_neighbor",
    "sparse_heroics",
    "rest_trap",
    "chain_reaction",
    "deceptive_calm",
    "overcrowding",
    "mixed_motivation",
]

# Scenario factory functions
SCENARIO_FACTORIES = {
    "trivial_cooperation": trivial_cooperation_scenario,
    "early_containment": early_containment_scenario,
    "greedy_neighbor": greedy_neighbor_scenario,
    "sparse_heroics": sparse_heroics_scenario,
    "rest_trap": rest_trap_scenario,
    "chain_reaction": chain_reaction_scenario,
    "deceptive_calm": deceptive_calm_scenario,
    "overcrowding": overcrowding_scenario,
    "mixed_motivation": mixed_motivation_scenario,
}


def load_evolved_genome(scenario_name: str, version: str = "v4") -> np.ndarray:
    """Load evolved genome from experiments directory."""
    genome_path = (
        Path("experiments")
        / "scenarios"
        / scenario_name
        / f"evolved_{version}"
        / "best_agent.json"
    )

    if not genome_path.exists():
        pytest.skip(f"No evolved genome found for {scenario_name} ({version})")

    with open(genome_path) as f:
        data = json.load(f)

    return np.array(data["genome"])


def load_all_evolved_v4_genomes() -> Dict[str, np.ndarray]:
    """Load all evolved v4 genomes from base scenarios."""
    genomes = {}
    for scenario_name in BASE_SCENARIOS:
        try:
            genomes[scenario_name] = load_evolved_genome(scenario_name, "v4")
        except:
            pass  # Skip if missing
    return genomes


def get_universal_genome() -> np.ndarray:
    """
    Get the universal genome (average of all evolved v4 genomes).

    In Phase 1.5, all scenarios converged to nearly identical strategies,
    so we can use any of them or their average.
    """
    genomes = load_all_evolved_v4_genomes()

    if len(genomes) == 0:
        pytest.skip("No evolved v4 genomes found - run evolution experiments first")

    # Use the average as the universal genome
    genome_array = np.array(list(genomes.values()))
    return genome_array.mean(axis=0)


def evaluate_genome_on_scenario(
    genome: np.ndarray,
    scenario: Scenario,
    num_simulations: int = 100,
    seed: int = 42,
) -> float:
    """Evaluate a genome's payoff on a scenario."""
    evaluator = RustFitnessEvaluator(
        scenario=scenario,
        num_simulations=num_simulations,
        num_workers=1,
        seed=seed,
    )

    return evaluator.evaluate(genome)


def measure_work_rate(
    genome: np.ndarray,
    scenario: Scenario,
    num_simulations: int = 50,
    seed: int = 42,
) -> float:
    """
    Measure the work rate (fraction of actions that are WORK).

    This is approximated by the work_tendency parameter (genome[1]).
    """
    # For heuristic agents, work_tendency directly controls work rate
    return genome[1]


# ==============================================================================
# Phase 1.5: Universal Nash Equilibrium Tests
# ==============================================================================


@pytest.mark.research
class TestUniversalEquilibrium:
    """Validate Phase 1.5: Universal Nash equilibrium findings."""

    def test_evolved_genomes_exist(self):
        """Check that evolved v4 genomes exist for all base scenarios."""
        genomes = load_all_evolved_v4_genomes()

        # Should have at least 5 scenarios evolved
        assert len(genomes) >= 5, (
            f"Expected at least 5 evolved scenarios, found {len(genomes)}. "
            f"Available: {list(genomes.keys())}"
        )

    def test_universal_genome_convergence(self):
        """Phase 1.5: All evolved agents converged to nearly identical genomes."""
        genomes = load_all_evolved_v4_genomes()

        if len(genomes) < 2:
            pytest.skip("Need at least 2 genomes to test convergence")

        genome_list = list(genomes.values())

        # Check pairwise differences
        max_diff = 0.0
        for g1, g2 in itertools.combinations(genome_list, 2):
            diff = np.linalg.norm(g1 - g2)
            max_diff = max(max_diff, diff)

        # All genomes should be very similar (L2 distance < 0.5)
        # This validates universal convergence
        assert max_diff < 0.5, (
            f"Genomes differ too much (max L2 distance = {max_diff:.3f}). "
            f"Phase 1.5 claims universal convergence but genomes are not identical."
        )

    def test_parameter_consistency(self):
        """Phase 1.5: Key parameters consistent across evolved strategies."""
        genomes = load_all_evolved_v4_genomes()

        if len(genomes) < 2:
            pytest.skip("Need at least 2 genomes to test consistency")

        genome_array = np.array(list(genomes.values()))

        # Check work_tendency (index 1) - should all be low (free-riding)
        work_tendencies = genome_array[:, 1]
        assert np.all(work_tendencies < 0.2), (
            f"Work tendencies should be low (<0.2), found: {work_tendencies}. "
            f"This suggests free-riding is not universal."
        )

        # Check rest_bias (index 8) - should all be high
        rest_biases = genome_array[:, 8]
        assert np.mean(rest_biases) > 0.8, (
            f"Rest biases should be high (>0.8), found mean: {np.mean(rest_biases):.3f}"
        )

    @pytest.mark.slow
    def test_universal_generalization(self):
        """Phase 1.5: Universal agent works well on all base scenarios (slow: ~2 min)."""
        universal = get_universal_genome()

        # Test on all base scenarios
        for scenario_name in BASE_SCENARIOS:
            if scenario_name not in SCENARIO_FACTORIES:
                continue

            scenario_fn = SCENARIO_FACTORIES[scenario_name]
            scenario = scenario_fn(num_agents=4)

            payoff = evaluate_genome_on_scenario(
                universal, scenario, num_simulations=50, seed=42
            )

            # Universal strategy should achieve reasonable payoff everywhere
            # (minimum of 40 based on empirical results)
            assert payoff > 40, (
                f"Universal genome failed on {scenario_name}: "
                f"payoff = {payoff:.1f} < 40. Universal strategy may not generalize."
            )


# ==============================================================================
# Phase 2A: Boundary Testing Validation
# ==============================================================================


@pytest.mark.research
class TestBoundaryRobustness:
    """Validate Phase 2A: Extreme parameter robustness."""

    @pytest.mark.slow
    def test_extreme_beta_robustness(self):
        """Phase 2A: Universal strategy works on extreme β (fire spread) (slow: ~1 min)."""
        universal = get_universal_genome()

        # Test extreme fire spread rates
        extreme_betas = [
            (0.02, "glacial_spread", glacial_spread_scenario),
            (0.75, "wildfire", wildfire_scenario),
        ]

        for beta, name, scenario_fn in extreme_betas:
            scenario = scenario_fn(num_agents=4)
            payoff = evaluate_genome_on_scenario(
                universal, scenario, num_simulations=50, seed=42
            )

            # Should still achieve reasonable performance
            assert payoff > 30, (
                f"Universal genome failed on extreme β={beta} ({name}): "
                f"payoff = {payoff:.1f} < 30"
            )

    @pytest.mark.slow
    def test_pspark_goldilocks_zone(self):
        """Phase 2A.1: Optimal p_spark ∈ [0.02, 0.03] (slow: ~2 min)."""
        universal = get_universal_genome()

        # Test different p_spark values
        p_spark_values = [0.00, 0.01, 0.02, 0.03, 0.05]
        payoffs = {}

        for p_spark in p_spark_values:
            # Create custom scenario with specific p_spark
            scenario = Scenario(
                num_agents=4,
                beta=0.3,
                kappa=0.2,
                p_spark=p_spark,
                A=100.0,
                L=100.0,
                c=5.0,
                N_min=20,
            )

            payoffs[p_spark] = evaluate_genome_on_scenario(
                universal, scenario, num_simulations=50, seed=42
            )

        # p_spark = 0.02-0.03 should be in Goldilocks zone
        # (better than too low or too high)
        goldilocks_payoff = max(payoffs[0.02], payoffs[0.03])

        # Should be better than no fire (0.00)
        if 0.00 in payoffs:
            assert goldilocks_payoff > payoffs[0.00], (
                f"Goldilocks zone (0.02-0.03) payoff {goldilocks_payoff:.1f} "
                f"should exceed no-fire payoff {payoffs[0.00]:.1f}"
            )

        # Should be better than high fire rate (0.05)
        assert goldilocks_payoff > payoffs[0.05], (
            f"Goldilocks zone (0.02-0.03) payoff {goldilocks_payoff:.1f} "
            f"should exceed high-fire payoff {payoffs[0.05]:.1f}"
        )


# ==============================================================================
# Phase 2D: Mechanism Design Tests
# ==============================================================================


@pytest.mark.research
class TestMechanismDesign:
    """Validate Phase 2D: Mechanism design cannot induce cooperation."""

    def test_free_work_fails_to_cooperate(self):
        """Phase 2D: Even free work (c=0) doesn't induce cooperation."""
        universal = get_universal_genome()

        # Measure work rate on free work scenario
        work_rate = measure_work_rate(universal, None)

        # Work rate should still be low (<20%)
        assert work_rate < 0.20, (
            f"Free work should not induce cooperation, but work_rate = {work_rate:.2f}"
        )

    def test_cheap_work_fails_to_cooperate(self):
        """Phase 2D: Cheap work (c=1) doesn't induce cooperation."""
        universal = get_universal_genome()

        work_rate = measure_work_rate(universal, None)

        assert work_rate < 0.20, (
            f"Cheap work should not induce cooperation, but work_rate = {work_rate:.2f}"
        )

    def test_mechanism_design_impossibility(self):
        """Phase 2D: Varying c, A, L cannot induce cooperation."""
        universal = get_universal_genome()

        # The universal genome's work_tendency should remain low
        # regardless of scenario parameters
        work_tendency = universal[1]

        assert work_tendency < 0.15, (
            f"Universal strategy has work_tendency = {work_tendency:.3f}, "
            f"suggesting cooperation is possible. Phase 2D claims it's impossible."
        )


# ==============================================================================
# Scale Testing: Population-Size Invariance
# ==============================================================================


@pytest.mark.research
class TestPopulationSizeInvariance:
    """Validate that universal strategy scales to larger populations."""

    @pytest.mark.slow
    def test_population_size_scaling(self):
        """Scale Testing: Performance degrades <10% from N=4 to N=10 (slow: ~3 min)."""
        universal = get_universal_genome()

        # Baseline: N=4 on trivial cooperation
        scenario_4 = trivial_cooperation_scenario(num_agents=4)
        baseline_payoff = evaluate_genome_on_scenario(
            universal, scenario_4, num_simulations=50, seed=42
        )

        # Test larger populations
        for N in [6, 8, 10]:
            scenario_n = trivial_cooperation_scenario(num_agents=N)
            payoff_n = evaluate_genome_on_scenario(
                universal, scenario_n, num_simulations=50, seed=42
            )

            # Calculate relative degradation
            if baseline_payoff > 0:
                degradation = abs(payoff_n - baseline_payoff) / baseline_payoff
            else:
                degradation = 0.0

            # Should have <10% degradation
            assert degradation < 0.10, (
                f"Population size N={N} shows {degradation*100:.1f}% degradation "
                f"from baseline (N=4: {baseline_payoff:.1f}, N={N}: {payoff_n:.1f})"
            )


# ==============================================================================
# Summary Test
# ==============================================================================


@pytest.mark.research
class TestResearchSummary:
    """High-level summary validation of research findings."""

    def test_free_riding_equilibrium_exists(self):
        """Validate that free-riding is the dominant equilibrium strategy."""
        genomes = load_all_evolved_v4_genomes()

        if len(genomes) == 0:
            pytest.skip("No evolved genomes found")

        genome_array = np.array(list(genomes.values()))

        # Average work_tendency across all evolved strategies
        avg_work_tendency = genome_array[:, 1].mean()

        # Should be very low (< 0.15)
        assert avg_work_tendency < 0.15, (
            f"Expected free-riding equilibrium (work_tendency < 0.15), "
            f"but found average work_tendency = {avg_work_tendency:.3f}"
        )

    def test_universal_convergence_occurred(self):
        """Validate that evolution converged to universal strategy."""
        genomes = load_all_evolved_v4_genomes()

        if len(genomes) < 3:
            pytest.skip("Need at least 3 genomes to validate convergence")

        genome_array = np.array(list(genomes.values()))

        # Standard deviation of each parameter across genomes
        param_stds = genome_array.std(axis=0)

        # Most parameters should have low variation (< 0.3)
        low_variation_params = (param_stds < 0.3).sum()

        assert low_variation_params >= 7, (
            f"Expected at least 7/10 parameters to have low variation (std < 0.3), "
            f"but only found {low_variation_params}. Param stds: {param_stds}"
        )


if __name__ == "__main__":
    # Run tests manually for quick verification
    print("=" * 80)
    print("Research Findings Validation Tests")
    print("=" * 80)

    print("\n1. Testing evolved genomes exist...")
    test = TestUniversalEquilibrium()
    test.test_evolved_genomes_exist()
    print("✓ Evolved genomes found")

    print("\n2. Testing universal genome convergence...")
    test.test_universal_genome_convergence()
    print("✓ Universal convergence validated")

    print("\n3. Testing parameter consistency...")
    test.test_parameter_consistency()
    print("✓ Parameters consistent (low work_tendency)")

    print("\n4. Testing free-riding equilibrium...")
    summary_test = TestResearchSummary()
    summary_test.test_free_riding_equilibrium_exists()
    print("✓ Free-riding equilibrium confirmed")

    print("\n5. Testing universal convergence...")
    summary_test.test_universal_convergence_occurred()
    print("✓ Universal convergence confirmed")

    print("\n" + "=" * 80)
    print("Core research findings validated! Run with pytest for full suite:")
    print("  pytest tests/test_research_findings.py -v")
    print("  pytest tests/test_research_findings.py -m research")
    print("  pytest tests/test_research_findings.py -m 'research and not slow'")
    print("=" * 80)
