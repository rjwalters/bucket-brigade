"""
Pytest configuration and fixtures.

This module provides:
- Custom pytest markers for test organization
- Dependency checking and auto-skip for missing dependencies
- Shared fixtures for common test objects
"""

import pytest
import sys


def pytest_configure(config):
    """Register custom markers."""
    # Markers are also defined in pyproject.toml, but we register them here
    # for better IDE support and documentation
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "research: marks tests validating research findings (deselect with '-m \"not research\"')",
    )
    config.addinivalue_line(
        "markers",
        "torch_required: marks tests requiring torch/RL dependencies",
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests with missing dependencies.

    This function checks for optional dependencies and automatically skips
    tests that require them if they're not installed.
    """
    # Check for torch availability
    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False

    if not torch_available:
        skip_torch = pytest.mark.skip(
            reason="torch not available (install with: uv sync --extra rl)"
        )
        for item in items:
            if "torch_required" in item.keywords:
                item.add_marker(skip_torch)

    # Check for proptest (Rust property testing)
    # This is for documentation - Rust tests are separate
    # But we could add Python property testing with hypothesis in the future

    # Mark slow tests with timing estimate if verbose
    if config.option.verbose >= 2:
        for item in items:
            if "slow" in item.keywords:
                # Try to extract timing from docstring
                if item.obj.__doc__ and "slow:" in item.obj.__doc__.lower():
                    item.add_marker(pytest.mark.filterwarnings("default"))


# ==============================================================================
# Shared Fixtures
# ==============================================================================


@pytest.fixture
def trivial_scenario():
    """Create a trivial cooperation scenario for testing."""
    from bucket_brigade.envs.scenarios import trivial_cooperation_scenario

    return trivial_cooperation_scenario(num_agents=4)


@pytest.fixture
def easy_scenario():
    """Create an easy scenario for testing."""
    from bucket_brigade.envs.scenarios import easy_scenario

    return easy_scenario(num_agents=4)


@pytest.fixture
def rust_evaluator(easy_scenario):
    """Create a Rust fitness evaluator for testing."""
    from bucket_brigade.evolution.fitness_rust import RustFitnessEvaluator

    return RustFitnessEvaluator(
        scenario=easy_scenario,
        num_simulations=10,
        num_workers=1,
        seed=42,
    )


@pytest.fixture
def payoff_evaluator(trivial_scenario):
    """Create a payoff evaluator for Nash equilibrium tests."""
    from bucket_brigade.equilibrium.payoff_evaluator import PayoffEvaluator

    return PayoffEvaluator(
        scenario=trivial_scenario,
        num_simulations=10,
        seed=42,
        parallel=False,
    )


@pytest.fixture
def firefighter_params():
    """Return firefighter archetype parameters."""
    from bucket_brigade.agents.archetypes import FIREFIGHTER_PARAMS

    return FIREFIGHTER_PARAMS.copy()


@pytest.fixture
def free_rider_params():
    """Return free rider archetype parameters."""
    from bucket_brigade.agents.archetypes import FREE_RIDER_PARAMS

    return FREE_RIDER_PARAMS.copy()


@pytest.fixture
def random_genome(seed=42):
    """Generate a random valid genome for testing."""
    import numpy as np

    rng = np.random.RandomState(seed)
    return rng.random(10)


# ==============================================================================
# Test Execution Helpers
# ==============================================================================


def pytest_report_header(config):
    """Add custom information to pytest header."""
    import bucket_brigade_core

    headers = []
    headers.append(f"bucket-brigade-core available: Yes")

    # Check optional dependencies
    try:
        import torch

        headers.append(f"torch (RL): {torch.__version__}")
    except ImportError:
        headers.append("torch (RL): Not installed")

    try:
        import pufferlib

        headers.append(f"pufferlib: {pufferlib.__version__}")
    except ImportError:
        headers.append("pufferlib: Not installed")

    return headers


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests (default: skip slow tests)",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (default: skip integration tests)",
    )


def pytest_runtest_setup(item):
    """Custom test setup based on markers and command-line options."""
    # Skip slow tests unless --run-slow is passed
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run slow tests")

    # Skip integration tests unless --run-integration is passed
    if "integration" in item.keywords and not item.config.getoption("--run-integration"):
        pytest.skip("need --run-integration option to run integration tests")
