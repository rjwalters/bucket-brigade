"""
Pytest configuration and fixtures.

This module provides:
- Custom pytest markers for test organization
- Dependency checking and auto-skip for missing dependencies
- Shared fixtures for common test objects
"""

import os
from pathlib import Path

import pytest


# Env var that converts a missing Rust extension from a hard collection
# failure into an explicit skip for `requires_rust` tests. Intended for the
# rare Python-only test run (e.g. on a machine without a Rust toolchain).
RUST_ESCAPE_HATCH_ENV = "BUCKET_BRIGADE_ALLOW_MISSING_RUST"

# Env var that converts a stale Rust extension (`.so` older than its Rust
# sources) from a hard collection failure into an explicit skip. Intended
# for CI / dev workflows that knowingly skip the freshness check (e.g. when
# file mtimes are unreliable post-checkout).
RUST_STALE_ESCAPE_HATCH_ENV = "BUCKET_BRIGADE_ALLOW_STALE_RUST"

# Path the user is told to run to fix a missing or stale Rust extension.
# The literal string is referenced in the UsageError below and asserted by
# tests; keep it in sync.
RUST_BUILD_SCRIPT = "bucket-brigade-core/build.sh"


def _rust_core_available() -> bool:
    """Return True iff ``bucket_brigade_core`` can be imported in this env."""
    try:
        import bucket_brigade_core  # noqa: F401
    except ImportError:
        return False
    return True


def _installed_rust_so_path() -> Path | None:
    """Locate the installed ``bucket_brigade_core`` ``.so`` artifact.

    Returns the filesystem path to the compiled PyO3 extension (the
    ``bucket_brigade_core.cpython-*.so`` file), or ``None`` if it cannot be
    found. The extension is built alongside the ``__init__.py`` package, so
    we resolve via ``bucket_brigade_core.__file__`` and glob for the ``.so``.
    """
    try:
        import bucket_brigade_core
    except ImportError:
        return None

    init_file = getattr(bucket_brigade_core, "__file__", None)
    if not init_file:
        return None

    pkg_dir = Path(init_file).parent
    candidates = sorted(pkg_dir.glob("bucket_brigade_core*.so"))
    return candidates[0] if candidates else None


def _rust_source_root() -> Path | None:
    """Locate the ``bucket-brigade-core/`` source root.

    Walks up from this conftest file looking for a sibling
    ``bucket-brigade-core/src`` directory. Returns ``None`` if not found
    (e.g. when running from an installed wheel without source checkout).
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "bucket-brigade-core"
        if (candidate / "src").is_dir():
            return candidate
    return None


def _rust_core_is_stale() -> bool:
    """Return True iff the installed Rust ``.so`` is older than its sources.

    Compares the mtime of the compiled extension against the newest mtime
    among ``bucket-brigade-core/src/**/*.rs`` and ``bucket-brigade-core/
    Cargo.toml``. Returns ``False`` (not stale) in any case where the check
    cannot be performed conservatively:

      - Extension not importable (handled by the absence check, not here).
      - Source tree not found (e.g. installed wheel, no source checkout).
      - No Rust source files discovered (defensive — would otherwise treat
        every install as fresh by default, which is correct).
    """
    so_path = _installed_rust_so_path()
    if so_path is None or not so_path.exists():
        # Absence is handled separately; don't double-trigger here.
        return False

    source_root = _rust_source_root()
    if source_root is None:
        return False

    sources: list[Path] = list((source_root / "src").rglob("*.rs"))
    cargo_toml = source_root / "Cargo.toml"
    if cargo_toml.exists():
        sources.append(cargo_toml)

    if not sources:
        return False

    try:
        so_mtime = so_path.stat().st_mtime
        newest_source_mtime = max(p.stat().st_mtime for p in sources)
    except OSError:
        # Filesystem error reading mtimes — fail open (don't block tests).
        return False

    return newest_source_mtime > so_mtime


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
    # `requires_rust` handling — see issue #189 (absence) and #330 (staleness).
    #
    # Behaviour:
    #   - Rust importable + fresh                    -> run normally.
    #   - Rust missing AND escape-hatch env unset    -> UsageError (loud).
    #   - Rust missing AND escape-hatch env set      -> explicit skip with
    #     a message naming the env var (NOT a silent skip).
    #   - Rust present but STALE AND staleness env   -> UsageError (loud).
    #     unset
    #   - Rust present but STALE AND staleness env   -> explicit skip with
    #     set                                          a message naming the
    #                                                  env var.
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
    elif requires_rust_items and _rust_core_is_stale():
        # Extension is importable but the .so is older than the Rust sources —
        # the common case after rebasing onto main commits that touched
        # bucket-brigade-core/src/. See issue #330.
        stale_escape_hatch = os.environ.get(RUST_STALE_ESCAPE_HATCH_ENV, "").strip()
        if stale_escape_hatch in {"", "0", "false", "False"}:
            raise pytest.UsageError(
                "bucket_brigade_core (Rust extension) is older than its source "
                f"tree, but {len(requires_rust_items)} test(s) marked "
                "@pytest.mark.requires_rust were collected. The installed "
                "extension will not reflect recent changes to "
                "bucket-brigade-core/src/ or Cargo.toml.\n"
                f"  Fix: run `bash {RUST_BUILD_SCRIPT}` to rebuild & reinstall the "
                "extension into the active venv.\n"
                f"  Escape hatch: set {RUST_STALE_ESCAPE_HATCH_ENV}=1 to skip the "
                "staleness check (intended for CI/dev workflows where mtime "
                "ordering is unreliable, e.g. after a fresh git checkout)."
            )
        # Escape hatch: silently accept stale extension — do NOT skip tests
        # (a stale extension may still pass, and the user opted in explicitly).

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
    rust_available = _rust_core_available()
    rust_present = "Yes" if rust_available else "No"
    headers.append(f"bucket-brigade-core available: {rust_present}")
    if rust_available:
        rust_stale = "Yes" if _rust_core_is_stale() else "No"
        headers.append(f"bucket-brigade-core stale: {rust_stale}")

    # Check optional dependencies
    try:
        import torch

        headers.append(f"torch (RL): {torch.__version__}")
    except ImportError:
        headers.append("torch (RL): Not installed")

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
