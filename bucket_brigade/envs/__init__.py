"""
Environment implementations for Bucket Brigade.
"""

from .bucket_brigade_env import BucketBrigadeEnv
from .macro_action_env import (
    MacroActionEnv,
    OPT_DEFEND_OWN,
    OPT_FOLLOW_BASE,
    OPT_PATROL,
    OPT_REST_UNTIL_FIRE,
)
from .scenarios_generated import (
    Scenario,
    default_scenario,
    easy_scenario,
    hard_scenario,
    # Named test scenarios
    trivial_cooperation_scenario,
    early_containment_scenario,
    greedy_neighbor_scenario,
    sparse_heroics_scenario,
    rest_trap_scenario,
    chain_reaction_scenario,
    deceptive_calm_scenario,
    overcrowding_scenario,
    mixed_motivation_scenario,
    # Scenario registry
    get_scenario_by_name,
    list_scenarios,
    SCENARIO_REGISTRY,
)
from .scenarios_random import random_scenario
from .registry import (
    DEFAULT_NUM_AGENTS,
    SCENARIO_VERSIONS,
    get_scenario_by_id,
    list_versioned_scenarios,
    parse_scenario_id,
)

__all__ = [
    "BucketBrigadeEnv",
    "MacroActionEnv",
    "OPT_PATROL",
    "OPT_DEFEND_OWN",
    "OPT_REST_UNTIL_FIRE",
    "OPT_FOLLOW_BASE",
    "Scenario",
    "default_scenario",
    "easy_scenario",
    "hard_scenario",
    "random_scenario",
    # Named test scenarios
    "trivial_cooperation_scenario",
    "early_containment_scenario",
    "greedy_neighbor_scenario",
    "sparse_heroics_scenario",
    "rest_trap_scenario",
    "chain_reaction_scenario",
    "deceptive_calm_scenario",
    "overcrowding_scenario",
    "mixed_motivation_scenario",
    # Scenario registry
    "get_scenario_by_name",
    "list_scenarios",
    "SCENARIO_REGISTRY",
    # Versioned scenario registry (issue #369)
    "DEFAULT_NUM_AGENTS",
    "SCENARIO_VERSIONS",
    "get_scenario_by_id",
    "list_versioned_scenarios",
    "parse_scenario_id",
]
