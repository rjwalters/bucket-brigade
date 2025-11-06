"""
Scenario generation and parameter management for Bucket Brigade environments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Scenario:
    """Represents the stochastic configuration of a game."""

    # Fire spread and extinguishing parameters
    beta: float  # Fire spread probability per neighbor
    kappa: float  # Extinguish efficiency

    # Reward parameters
    A: float  # Reward per saved house
    L: float  # Penalty per ruined house
    c: float  # Cost per worker per night

    # Initial conditions
    rho_ignite: float  # Initial fraction of houses burning
    N_min: int  # Minimum nights before termination
    p_spark: float  # Probability of spontaneous ignition
    N_spark: int  # Number of nights with sparks active

    # Game setup
    num_agents: int  # Number of agents participating

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert scenario parameters to a feature vector for agent conditioning.
        Returns a numpy array with scenario features.
        """
        return np.array(
            [
                self.beta,
                self.kappa,
                self.A,
                self.L,
                self.c,
                self.rho_ignite,
                self.N_min,
                self.p_spark,
                self.N_spark,
                self.num_agents,
            ],
            dtype=np.float32,
        )


def random_scenario(num_agents: int, seed: Optional[int] = None) -> Scenario:
    """
    Generate a random scenario with parameters sampled from reasonable distributions.

    Args:
        num_agents: Number of agents in the game
        seed: Random seed for reproducibility

    Returns:
        A randomly sampled Scenario
    """
    if seed is not None:
        np.random.seed(seed)

    return Scenario(
        # Fire spread: moderate probability
        beta=np.random.uniform(0.15, 0.35),
        # Extinguish efficiency: moderate effectiveness
        kappa=np.random.uniform(0.4, 0.6),
        # Rewards: balanced incentives
        A=100.0,  # Reward per saved house
        L=100.0,  # Penalty per ruined house
        c=0.5,  # Cost per worker per night
        # Initial burning fraction: some fires to start
        rho_ignite=np.random.uniform(0.1, 0.3),
        # Minimum nights: reasonable game length
        N_min=np.random.randint(10, 20),
        # Spontaneous ignition: sometimes active
        p_spark=np.random.choice([0.0, np.random.uniform(0.01, 0.05)]),
        N_spark=0,  # Will be set to N_min if sparks are active
        num_agents=num_agents,
    )


def default_scenario(num_agents: int) -> Scenario:
    """
    Create a default scenario with reasonable fixed parameters.

    Args:
        num_agents: Number of agents in the game

    Returns:
        A default Scenario with typical parameters
    """
    return Scenario(
        beta=0.25,
        kappa=0.5,
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.2,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def easy_scenario(num_agents: int) -> Scenario:
    """
    Create an easy scenario with low fire spread and high extinguish efficiency.

    Args:
        num_agents: Number of agents in the game

    Returns:
        An easy Scenario
    """
    return Scenario(
        beta=0.1,
        kappa=0.8,
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=10,
        p_spark=0.01,
        N_spark=10,
        num_agents=num_agents,
    )


def hard_scenario(num_agents: int) -> Scenario:
    """
    Create a hard scenario with high fire spread and low extinguish efficiency.

    Args:
        num_agents: Number of agents in the game

    Returns:
        A hard Scenario
    """
    return Scenario(
        beta=0.4,
        kappa=0.3,
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.3,
        N_min=15,
        p_spark=0.05,
        N_spark=15,
        num_agents=num_agents,
    )


# Named scenario distributions from SCENARIO_BRAINSTORM.md


def trivial_cooperation_scenario(num_agents: int) -> Scenario:
    """Scenario 1: Trivial Cooperation - fires are rare and extinguish easily."""
    return Scenario(
        beta=0.15,  # low spread
        kappa=0.9,  # high extinguish rate
        A=100.0,
        L=100.0,
        c=0.5,  # low work cost
        rho_ignite=0.1,  # few initial fires
        N_min=12,
        p_spark=0.0,  # no spontaneous fires
        N_spark=12,
        num_agents=num_agents,
    )


def early_containment_scenario(num_agents: int) -> Scenario:
    """Scenario 2: Early Containment - fires start aggressive but can be stopped early."""
    return Scenario(
        beta=0.35,  # high spread
        kappa=0.6,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.3,  # many initial fires
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def greedy_neighbor_scenario(num_agents: int) -> Scenario:
    """Scenario 3: Greedy Neighbor - social dilemma between self-interest and cooperation."""
    return Scenario(
        beta=0.15,  # low spread
        kappa=0.4,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=1.0,  # high work cost
        rho_ignite=0.2,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def sparse_heroics_scenario(num_agents: int) -> Scenario:
    """Scenario 4: Sparse Heroics - few workers can make the difference."""
    return Scenario(
        beta=0.1,  # very low spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.8,  # moderate-high work cost
        rho_ignite=0.15,
        N_min=20,  # long games
        p_spark=0.02,
        N_spark=20,
        num_agents=num_agents,
    )


def rest_trap_scenario(num_agents: int) -> Scenario:
    """Scenario 5: Rest Trap - fires usually extinguish themselves, but not always."""
    return Scenario(
        beta=0.05,  # very low spread
        kappa=0.95,  # very high extinguish rate
        A=100.0,
        L=100.0,
        c=0.2,  # low work cost
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.02,  # occasional sparks
        N_spark=12,
        num_agents=num_agents,
    )


def chain_reaction_scenario(num_agents: int) -> Scenario:
    """Scenario 6: Chain Reaction - high spread requires distributed teams."""
    return Scenario(
        beta=0.45,  # high spread
        kappa=0.6,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.7,  # moderate work cost
        rho_ignite=0.3,  # many initial fires
        N_min=15,
        p_spark=0.03,
        N_spark=15,
        num_agents=num_agents,
    )


def deceptive_calm_scenario(num_agents: int) -> Scenario:
    """Scenario 7: Deceptive Calm - occasional flare-ups reward honest signaling."""
    return Scenario(
        beta=0.25,  # moderate spread
        kappa=0.6,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.4,  # low-moderate work cost
        rho_ignite=0.1,  # few initial fires
        N_min=20,  # long games
        p_spark=0.05,  # occasional sparks
        N_spark=20,
        num_agents=num_agents,
    )


def overcrowding_scenario(num_agents: int) -> Scenario:
    """Scenario 8: Overcrowding - too many workers reduce efficiency."""
    return Scenario(
        beta=0.2,  # low spread
        kappa=0.3,  # low extinguish efficiency
        A=50.0,  # lower reward
        L=100.0,  # same penalty
        c=0.6,  # moderate work cost
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def mixed_motivation_scenario(num_agents: int) -> Scenario:
    """Scenario 10: Mixed Motivation - ownership creates self-interest conflicts."""
    return Scenario(
        beta=0.3,  # moderate spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.6,  # moderate work cost
        rho_ignite=0.2,
        N_min=15,
        p_spark=0.03,
        N_spark=15,
        num_agents=num_agents,
    )


# Phase 2A: Extreme scenarios for universality boundary testing


def glacial_spread_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Glacial Spread - fires barely spread (β=0.02)."""
    return Scenario(
        beta=0.02,  # extremely low spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.5,  # moderate work cost
        rho_ignite=0.15,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def explosive_spread_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Explosive Spread - fires spread very aggressively (β=0.60)."""
    return Scenario(
        beta=0.60,  # extremely high spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.5,  # moderate work cost
        rho_ignite=0.2,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def wildfire_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Wildfire - fires spread almost uncontrollably (β=0.75)."""
    return Scenario(
        beta=0.75,  # near-maximum spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.5,  # moderate work cost
        rho_ignite=0.25,
        N_min=10,
        p_spark=0.03,
        N_spark=10,
        num_agents=num_agents,
    )


def free_work_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Free Work - work costs almost nothing (c=0.05)."""
    return Scenario(
        beta=0.20,  # moderate spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.05,  # extremely low work cost
        rho_ignite=0.15,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def cheap_work_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Cheap Work - work is very affordable (c=0.10)."""
    return Scenario(
        beta=0.20,  # moderate spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.10,  # very low work cost
        rho_ignite=0.15,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def expensive_work_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Expensive Work - work is costly (c=2.0)."""
    return Scenario(
        beta=0.20,  # moderate spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=2.0,  # high work cost
        rho_ignite=0.15,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def prohibitive_work_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Prohibitive Work - work is extremely expensive (c=5.0)."""
    return Scenario(
        beta=0.20,  # moderate spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=5.0,  # extremely high work cost
        rho_ignite=0.15,
        N_min=12,
        p_spark=0.02,
        N_spark=12,
        num_agents=num_agents,
    )


def crisis_cheap_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Crisis + Cheap - fast spread but affordable work (β=0.60, c=0.10)."""
    return Scenario(
        beta=0.60,  # very high spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.10,  # very low work cost
        rho_ignite=0.25,
        N_min=10,
        p_spark=0.03,
        N_spark=10,
        num_agents=num_agents,
    )


def calm_expensive_scenario(num_agents: int) -> Scenario:
    """Phase 2A: Calm + Expensive - slow spread but costly work (β=0.02, c=5.0)."""
    return Scenario(
        beta=0.02,  # extremely low spread
        kappa=0.5,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=5.0,  # extremely high work cost
        rho_ignite=0.1,
        N_min=15,
        p_spark=0.01,
        N_spark=15,
        num_agents=num_agents,
    )


# Phase 2A.1: Trivial Cooperation Investigation
# Testing κ (extinguish rate) and p_spark (ongoing fire generation) to understand
# when universal strategy's over-cooperation becomes problematic


def easy_kappa_60_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with κ=0.60 (baseline)."""
    return Scenario(
        beta=0.15,  # low spread (same as trivial_cooperation)
        kappa=0.60,  # moderate-high extinguish
        A=100.0,
        L=100.0,
        c=0.5,  # moderate cost (same as trivial_cooperation)
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.0,  # no ongoing fires (same as trivial_cooperation)
        N_spark=12,
        num_agents=num_agents,
    )


def easy_kappa_70_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with κ=0.70."""
    return Scenario(
        beta=0.15,
        kappa=0.70,  # high extinguish
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.0,
        N_spark=12,
        num_agents=num_agents,
    )


def easy_kappa_80_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with κ=0.80."""
    return Scenario(
        beta=0.15,
        kappa=0.80,  # very high extinguish
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.0,
        N_spark=12,
        num_agents=num_agents,
    )


def easy_kappa_90_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with κ=0.90 (same as trivial_cooperation)."""
    return Scenario(
        beta=0.15,
        kappa=0.90,  # extremely high extinguish (trivial_cooperation value)
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.0,
        N_spark=12,
        num_agents=num_agents,
    )


def easy_spark_01_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with p_spark=0.01 (minimal ongoing fires)."""
    return Scenario(
        beta=0.15,
        kappa=0.90,  # keep high extinguish
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.01,  # minimal ongoing fires
        N_spark=12,
        num_agents=num_agents,
    )


def easy_spark_02_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with p_spark=0.02 (moderate ongoing fires)."""
    return Scenario(
        beta=0.15,
        kappa=0.90,
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.02,  # moderate ongoing fires
        N_spark=12,
        num_agents=num_agents,
    )


def easy_spark_05_scenario(num_agents: int) -> Scenario:
    """Phase 2A.1: Easy scenario with p_spark=0.05 (high ongoing fires)."""
    return Scenario(
        beta=0.15,
        kappa=0.90,
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=0.1,
        N_min=12,
        p_spark=0.05,  # high ongoing fires
        N_spark=12,
        num_agents=num_agents,
    )


# Phase 2D: Mechanism Design for Cooperation
# Testing scenario designs that attempt to induce cooperation
# and break the universal free-riding equilibrium


def nearly_free_work_scenario(num_agents: int) -> Scenario:
    """Phase 2D: Nearly Free Work - work cost approaching zero (c=0.01).

    Tests if free work breaks free-riding equilibrium.
    """
    return Scenario(
        beta=0.30,  # moderate spread
        kappa=0.60,  # moderate extinguish
        A=100.0,
        L=100.0,
        c=0.01,  # nearly free work (vs 0.05 in free_work)
        rho_ignite=0.15,
        N_min=12,
        p_spark=0.02,  # persistent threat
        N_spark=12,
        num_agents=num_agents,
    )


def front_loaded_crisis_scenario(num_agents: int) -> Scenario:
    """Phase 2D: Front-Loaded Crisis - overwhelming initial fires requiring immediate response.

    Many fires at start, spread explosively, but no ongoing fires.
    Tests crisis response vs sustained threat.
    """
    return Scenario(
        beta=0.70,  # very fast spread
        kappa=0.40,  # hard to extinguish
        A=100.0,
        L=100.0,
        c=0.30,  # affordable work
        rho_ignite=0.40,  # very high initial fires (vs 0.15-0.25 typical)
        N_min=8,  # fires spread easily
        p_spark=0.0,  # no ongoing fires - one-time crisis
        N_spark=12,
        num_agents=num_agents,
    )


def sustained_pressure_scenario(num_agents: int) -> Scenario:
    """Phase 2D: Sustained Pressure - continuous high threat requiring persistent effort.

    Very high ongoing fires with fast spread and difficult extinguishing.
    Tests if overwhelming sustained pressure induces cooperation.
    """
    return Scenario(
        beta=0.50,  # fast spread
        kappa=0.30,  # difficult to extinguish
        A=100.0,
        L=100.0,
        c=0.40,  # moderate cost
        rho_ignite=0.20,
        N_min=10,
        p_spark=0.10,  # very high ongoing fires (vs 0.02 typical)
        N_spark=8,  # easy spontaneous ignition
        num_agents=num_agents,
    )


def high_stakes_scenario(num_agents: int) -> Scenario:
    """Phase 2D: High Stakes - extreme asset values creating high-variance outcomes.

    Tests if increased payoff variance induces coordination.
    """
    return Scenario(
        beta=0.40,  # moderate-high spread
        kappa=0.50,  # moderate extinguish
        A=500.0,  # 5x normal asset value
        L=500.0,  # 5x normal loss
        c=1.0,  # work cost remains same (relatively cheaper)
        rho_ignite=0.20,
        N_min=12,
        p_spark=0.03,  # persistent threat
        N_spark=12,
        num_agents=num_agents,
    )


# Named distributions for random sampling


def sample_easy_coop_scenario(num_agents: int, seed: Optional[int] = None) -> Scenario:
    """Sample from Easy Cooperation distribution."""
    if seed is not None:
        np.random.seed(seed)
    return Scenario(
        beta=np.random.uniform(0.1, 0.2),
        kappa=np.random.uniform(0.7, 0.9),
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=np.random.uniform(0.05, 0.15),
        N_min=np.random.randint(10, 15),
        p_spark=0.0,
        N_spark=0,
        num_agents=num_agents,
    )


def sample_crisis_scenario(num_agents: int, seed: Optional[int] = None) -> Scenario:
    """Sample from Crisis distribution."""
    if seed is not None:
        np.random.seed(seed)
    return Scenario(
        beta=np.random.uniform(0.3, 0.5),
        kappa=np.random.uniform(0.4, 0.6),
        A=100.0,
        L=100.0,
        c=0.5,
        rho_ignite=np.random.uniform(0.2, 0.4),
        N_min=np.random.randint(12, 18),
        p_spark=np.random.uniform(0.02, 0.04),
        N_spark=0,  # Will be set to N_min
        num_agents=num_agents,
    )


def sample_sparse_work_scenario(
    num_agents: int, seed: Optional[int] = None
) -> Scenario:
    """Sample from Sparse Work distribution."""
    if seed is not None:
        np.random.seed(seed)
    return Scenario(
        beta=np.random.uniform(0.1, 0.2),
        kappa=np.random.uniform(0.4, 0.6),
        A=100.0,
        L=100.0,
        c=np.random.uniform(0.6, 0.9),
        rho_ignite=np.random.uniform(0.1, 0.2),
        N_min=np.random.randint(15, 25),
        p_spark=np.random.uniform(0.01, 0.03),
        N_spark=0,  # Will be set to N_min
        num_agents=num_agents,
    )


def sample_deception_scenario(num_agents: int, seed: Optional[int] = None) -> Scenario:
    """Sample from Deception distribution."""
    if seed is not None:
        np.random.seed(seed)
    return Scenario(
        beta=0.25,
        kappa=0.5,
        A=100.0,
        L=100.0,
        c=0.6,
        rho_ignite=0.2,
        N_min=15,
        p_spark=np.random.uniform(0.03, 0.05),
        N_spark=15,
        num_agents=num_agents,
    )


# Scenario Registry

SCENARIO_REGISTRY = {
    "default": default_scenario,
    "easy": easy_scenario,
    "hard": hard_scenario,
    "trivial_cooperation": trivial_cooperation_scenario,
    "early_containment": early_containment_scenario,
    "greedy_neighbor": greedy_neighbor_scenario,
    "sparse_heroics": sparse_heroics_scenario,
    "rest_trap": rest_trap_scenario,
    "chain_reaction": chain_reaction_scenario,
    "deceptive_calm": deceptive_calm_scenario,
    "overcrowding": overcrowding_scenario,
    "mixed_motivation": mixed_motivation_scenario,
    # Phase 2A: Extreme scenarios for boundary testing
    "glacial_spread": glacial_spread_scenario,
    "explosive_spread": explosive_spread_scenario,
    "wildfire": wildfire_scenario,
    "free_work": free_work_scenario,
    "cheap_work": cheap_work_scenario,
    "expensive_work": expensive_work_scenario,
    "prohibitive_work": prohibitive_work_scenario,
    "crisis_cheap": crisis_cheap_scenario,
    "calm_expensive": calm_expensive_scenario,
    # Phase 2A.1: Trivial cooperation investigation (κ and p_spark sweeps)
    "easy_kappa_60": easy_kappa_60_scenario,
    "easy_kappa_70": easy_kappa_70_scenario,
    "easy_kappa_80": easy_kappa_80_scenario,
    "easy_kappa_90": easy_kappa_90_scenario,
    "easy_spark_01": easy_spark_01_scenario,
    "easy_spark_02": easy_spark_02_scenario,
    "easy_spark_05": easy_spark_05_scenario,
    # Phase 2D: Mechanism design for cooperation
    "nearly_free_work": nearly_free_work_scenario,
    "front_loaded_crisis": front_loaded_crisis_scenario,
    "sustained_pressure": sustained_pressure_scenario,
    "high_stakes": high_stakes_scenario,
}


def get_scenario_by_name(name: str, num_agents: int) -> Scenario:
    """
    Get a scenario by name.

    Args:
        name: Scenario name (e.g., "trivial_cooperation", "default")
        num_agents: Number of agents in the scenario

    Returns:
        Scenario object

    Raises:
        ValueError: If scenario name is invalid

    Example:
        >>> scenario = get_scenario_by_name("trivial_cooperation", num_agents=4)
        >>> env = PufferBucketBrigade(scenario=scenario)
    """
    if name not in SCENARIO_REGISTRY:
        valid_names = ", ".join(sorted(SCENARIO_REGISTRY.keys()))
        raise ValueError(f"Unknown scenario '{name}'. Valid options: {valid_names}")

    return SCENARIO_REGISTRY[name](num_agents)


def list_scenarios() -> list:
    """
    Get list of available scenario names.

    Returns:
        Sorted list of scenario names

    Example:
        >>> scenarios = list_scenarios()
        >>> print(scenarios)
        ['chain_reaction', 'deceptive_calm', 'default', ...]
    """
    return sorted(SCENARIO_REGISTRY.keys())
