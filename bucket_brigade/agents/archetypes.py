"""
Archetypal strategy parameter vectors for Bucket Brigade heuristic agents.

These are well-known behavioral profiles that capture different playstyles.
Parameter order: [honesty_bias, work_tendency, neighbor_help_bias, own_house_priority,
                  risk_aversion, coordination_weight, exploration_rate, fatigue_memory,
                  rest_reward_bias, altruism_factor]
"""

import numpy as np

# Firefighter: Honest, hard-working, cooperative agent
FIREFIGHTER_PARAMS = np.array([
    1.0,  # honesty_bias - always truthful
    0.9,  # work_tendency - works most nights
    0.5,  # neighbor_help_bias - balanced
    0.8,  # own_house_priority - prioritizes own house
    0.5,  # risk_aversion - moderate
    0.7,  # coordination_weight - trusts signals
    0.1,  # exploration_rate - low randomness
    0.0,  # fatigue_memory - no inertia
    0.0,  # rest_reward_bias - doesn't prefer rest
    0.8,  # altruism_factor - high cooperation
])

# Free Rider: Selfish agent that avoids work
FREE_RIDER_PARAMS = np.array([
    0.7,  # honesty_bias - mostly truthful
    0.2,  # work_tendency - avoids work
    0.0,  # neighbor_help_bias - doesn't help neighbors
    0.9,  # own_house_priority - only cares about own house
    0.0,  # risk_aversion - not concerned with community fires
    0.0,  # coordination_weight - ignores signals
    0.1,  # exploration_rate - low randomness
    0.0,  # fatigue_memory - no inertia
    0.9,  # rest_reward_bias - strongly prefers rest
    0.0,  # altruism_factor - no altruism
])

# Hero: Maximum effort, maximum cooperation
HERO_PARAMS = np.array([
    1.0,  # honesty_bias - always truthful
    1.0,  # work_tendency - always works
    1.0,  # neighbor_help_bias - helps everyone
    0.5,  # own_house_priority - balanced
    0.1,  # risk_aversion - brave
    0.5,  # coordination_weight - moderate trust
    0.0,  # exploration_rate - no randomness
    0.9,  # fatigue_memory - consistent behavior
    0.0,  # rest_reward_bias - never rests
    1.0,  # altruism_factor - maximum altruism
])

# Coordinator: Balanced, trust-based strategy
COORDINATOR_PARAMS = np.array([
    0.9,  # honesty_bias - mostly truthful
    0.6,  # work_tendency - moderate work
    0.7,  # neighbor_help_bias - cooperative
    0.6,  # own_house_priority - balanced
    0.8,  # risk_aversion - cautious
    1.0,  # coordination_weight - high trust in signals
    0.05, # exploration_rate - very low randomness
    0.0,  # fatigue_memory - no inertia
    0.2,  # rest_reward_bias - slight rest preference
    0.6,  # altruism_factor - moderate altruism
])

# Liar: Deceptive agent with selfish motives
LIAR_PARAMS = np.array([
    0.1,  # honesty_bias - mostly dishonest
    0.7,  # work_tendency - works when beneficial
    0.0,  # neighbor_help_bias - no neighbor help
    0.9,  # own_house_priority - highly selfish
    0.2,  # risk_aversion - moderate
    0.8,  # coordination_weight - reads signals but lies
    0.3,  # exploration_rate - moderate randomness
    0.0,  # fatigue_memory - no inertia
    0.4,  # rest_reward_bias - moderate rest preference
    0.2,  # altruism_factor - low altruism
])

# Registry of archetypes for easy access
ARCHETYPES = {
    "firefighter": FIREFIGHTER_PARAMS,
    "free_rider": FREE_RIDER_PARAMS,
    "hero": HERO_PARAMS,
    "coordinator": COORDINATOR_PARAMS,
    "liar": LIAR_PARAMS,
}


def get_archetype(name: str) -> np.ndarray:
    """
    Get archetype parameter vector by name.

    Args:
        name: Archetype name (case-insensitive)

    Returns:
        10-dimensional parameter vector

    Raises:
        ValueError: If archetype name not found
    """
    name_lower = name.lower()
    if name_lower not in ARCHETYPES:
        valid_names = ", ".join(ARCHETYPES.keys())
        raise ValueError(f"Unknown archetype '{name}'. Valid options: {valid_names}")

    return ARCHETYPES[name_lower].copy()


def list_archetypes() -> list[str]:
    """Get list of available archetype names."""
    return sorted(ARCHETYPES.keys())
