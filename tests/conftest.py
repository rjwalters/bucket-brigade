"""
Pytest configuration and fixtures.

This module provides:
- Custom pytest markers for test organization
- Dependency checking and auto-skip for missing dependencies
- Shared fixtures for common test objects
"""

import os

import pytest


# Env var that converts a missing Rust extension from a hard collection
# failure into an explicit skip for `requires_rust` tests. Intended for the
# rare Python-only test run (e.g. on a machine without a Rust toolchain).
RUST_ESCAPE_HATCH_ENV = "BUCKET_BRIGADE_ALLOW_MISSING_RUST"

# Path the user is told to run to fix a missing Rust extension. The literal
# string is referenced in the UsageError below and asserted by tests; keep
# it in sync.
RUST_BUILD_SCRIPT = "bucket-brigade-core/build.sh"


def _rust_core_available() -> bool:
    """Return True iff ``bucket_brigade_core`` can be imported in this env."""
    try:
        import bucket_brigade_core  # noqa: F401
    except ImportError:
        return False
    return True


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
    config.addinivalue_line(
        "markers",
        "requires_rust: marks tests that require the bucket-brigade-core Rust extension to be built",
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests with missing dependencies, and surface a loud
    error (or explicit skip) when ``requires_rust`` tests are collected
    without the Rust extension importable.
    """
    # ------------------------------------------------------------------
    # `requires_rust` handling — see issue #189.
    #
    # Behaviour:
    #   - Rust importable                            -> run normally.
    #   - Rust missing AND escape-hatch env unset    -> UsageError (loud).
    #   - Rust missing AND escape-hatch env set      -> explicit skip with
    #     a message naming the env var (NOT a silent skip).
    # ------------------------------------------------------------------
    requires_rust_items = [item for item in items if "requires_rust" in item.keywords]

    if requires_rust_items and not _rust_core_available():
        escape_hatch = os.environ.get(RUST_ESCAPE_HATCH_ENV, "").strip()
        if escape_hatch in {"", "0", "false", "False"}:
            raise pytest.UsageError(
                "bucket_brigade_core (Rust extension) is not importable, but "
                f"{len(requires_rust_items)} test(s) marked @pytest.mark.requires_rust "
                "were collected.\n"
                f"  Fix: run `bash {RUST_BUILD_SCRIPT}` to build & install the extension "
                "into the active venv.\n"
                f"  Escape hatch: set {RUST_ESCAPE_HATCH_ENV}=1 to skip these tests "
                "instead of failing (intended for Python-only runs without a Rust "
                "toolchain)."
            )
        # Escape hatch: convert to explicit skips with a clear reason.
        skip_rust = pytest.mark.skip(
            reason=(
                f"Rust extension not built, skipping per {RUST_ESCAPE_HATCH_ENV}=1 "
                f"(build with `bash {RUST_BUILD_SCRIPT}` to run these tests)"
            )
        )
        for item in requires_rust_items:
            item.add_marker(skip_rust)

    # Check for torch availability
    try:
        import torch  # noqa: F401

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
    from bucket_brigade.envs import trivial_cooperation_scenario

    return trivial_cooperation_scenario(num_agents=4)


@pytest.fixture
def easy_scenario():
    """Create an easy scenario for testing."""
    from bucket_brigade.envs import easy_scenario

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

    headers = []
    rust_present = "Yes" if _rust_core_available() else "No"
    headers.append(f"bucket-brigade-core available: {rust_present}")

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
    if "integration" in item.keywords and not item.config.getoption(
        "--run-integration"
    ):
        pytest.skip("need --run-integration option to run integration tests")
