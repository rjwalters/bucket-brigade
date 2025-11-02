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
