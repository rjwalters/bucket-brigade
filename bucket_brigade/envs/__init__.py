"""
Environment implementations for Bucket Brigade.
"""

from typing import TYPE_CHECKING, Optional, Type, Callable, Any

from .bucket_brigade_env import BucketBrigadeEnv
from .macro_action_env import (
    MacroActionEnv,
    OPT_DEFEND_OWN,
    OPT_FOLLOW_BASE,
    OPT_PATROL,
    OPT_REST_UNTIL_FIRE,
)

# Optional PufferLib imports (only available if gymnasium is installed)
if TYPE_CHECKING:
    pass

try:
    # Use Rust-backed PufferLib environment for 100x speedup
    from .puffer_env_rust import (
        RustPufferBucketBrigade as PufferBucketBrigade,
        RustPufferBucketBrigadeVectorized as PufferBucketBrigadeVectorized,
        make_rust_env as make_env,
        make_rust_vectorized_env as make_vectorized_env,
    )
except ImportError:
    # PufferLib not available, skip these imports
    PufferBucketBrigade: Optional[Type[Any]] = None  # type: ignore[assignment, misc, no-redef]
    PufferBucketBrigadeVectorized: Optional[Type[Any]] = None  # type: ignore[assignment, misc, no-redef]
    make_env: Optional[Callable[..., Any]] = None  # type: ignore[assignment, misc, no-redef]
    make_vectorized_env: Optional[Callable[..., Any]] = None  # type: ignore[assignment, misc, no-redef]
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

__all__ = [
    "BucketBrigadeEnv",
    "MacroActionEnv",
    "OPT_PATROL",
    "OPT_DEFEND_OWN",
    "OPT_REST_UNTIL_FIRE",
    "OPT_FOLLOW_BASE",
    "PufferBucketBrigade",
    "PufferBucketBrigadeVectorized",
    "make_env",
    "make_vectorized_env",
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
]
