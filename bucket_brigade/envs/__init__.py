"""
Environment implementations for Bucket Brigade.
"""

from typing import TYPE_CHECKING, Optional, Type, Callable, Any

from .bucket_brigade_env import BucketBrigadeEnv

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
from .scenarios import (
    Scenario,
    random_scenario,
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
    # Sampling distributions
    sample_easy_coop_scenario,
    sample_crisis_scenario,
    sample_sparse_work_scenario,
    sample_deception_scenario,
    # Scenario registry
    get_scenario_by_name,
    list_scenarios,
    SCENARIO_REGISTRY,
)

__all__ = [
    "BucketBrigadeEnv",
    "PufferBucketBrigade",
    "PufferBucketBrigadeVectorized",
    "make_env",
    "make_vectorized_env",
    "Scenario",
    "random_scenario",
    "default_scenario",
    "easy_scenario",
    "hard_scenario",
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
    # Sampling distributions
    "sample_easy_coop_scenario",
    "sample_crisis_scenario",
    "sample_sparse_work_scenario",
    "sample_deception_scenario",
    # Scenario registry
    "get_scenario_by_name",
    "list_scenarios",
    "SCENARIO_REGISTRY",
]
