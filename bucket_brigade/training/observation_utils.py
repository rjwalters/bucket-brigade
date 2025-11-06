"""
Utilities for converting Rust observations to numpy arrays.
"""

import numpy as np
from bucket_brigade_core import AgentObservation


def flatten_observation(obs: AgentObservation, scenario_info: np.ndarray = None) -> np.ndarray:
    """
    Convert PyAgentObservation to flat numpy array for neural network input.

    Args:
        obs: PyAgentObservation from Rust environment
        scenario_info: Optional scenario parameters (10 values). If None, uses zeros.

    Returns:
        Flattened observation array

    Structure:
        - houses: [num_houses] float values (fire level 0.0-1.0)
        - signals: [num_agents] float values (signal from each agent)
        - locations: [num_agents] int values (house location of each agent)
        - last_actions: [num_agents * 2] values (house, mode for each agent)
        - scenario_info: [10] values (scenario parameters)
    """
    # Convert observation components to numpy arrays
    houses = np.array(obs.houses, dtype=np.float32)
    signals = np.array(obs.signals, dtype=np.float32)
    locations = np.array(obs.locations, dtype=np.float32)

    # Convert last actions (list of [house, mode] pairs) to flat array
    last_actions = np.array(obs.last_actions, dtype=np.float32).flatten()

    # Use provided scenario info or zeros
    if scenario_info is None:
        scenario_info = np.zeros(10, dtype=np.float32)
    else:
        scenario_info = np.array(scenario_info, dtype=np.float32)

    # Concatenate all components
    flat_obs = np.concatenate([
        houses,
        signals,
        locations,
        last_actions,
        scenario_info,
    ])

    return flat_obs


def get_observation_dim(num_houses: int, num_agents: int) -> int:
    """
    Calculate the observation dimension given game parameters.

    Args:
        num_houses: Number of houses in the game
        num_agents: Number of agents in the game

    Returns:
        Total observation dimension
    """
    return (
        num_houses +      # houses
        num_agents +      # signals
        num_agents +      # locations
        num_agents * 2 +  # last_actions (house + mode per agent)
        10                # scenario_info
    )


def create_scenario_info(scenario) -> np.ndarray:
    """
    Create scenario info array from scenario object.

    Args:
        scenario: PyScenario object

    Returns:
        10-element array with scenario parameters
    """
    return np.array([
        scenario.prob_fire_spreads_to_neighbor,
        scenario.prob_solo_agent_extinguishes_fire,
        scenario.prob_house_catches_fire,
        scenario.team_reward_house_survives,
        scenario.team_penalty_house_burns,
        scenario.cost_to_work_one_night,
        float(scenario.min_nights),
        0.0,  # Padding
        0.0,  # Padding
        0.0,  # Padding
    ], dtype=np.float32)
