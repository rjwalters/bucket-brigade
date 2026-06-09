"""Pip-wheel build smoke test (issues #373, #404).

Marked ``slow`` because it shells out to ``uv build`` and builds an
actual wheel; not part of the fast-test gate but is run by the
``-m "not slow"`` carve-out and by anyone validating a release.

Validates:

1. ``uv build --wheel`` succeeds from the repo root.
2. The produced ``.whl`` is non-empty and contains
   ``bucket_brigade/__init__.py``.
3. The wheel contains the ``baselines/release/local/`` placeholder
   (so the in-wheel artifact dir exists once #371 populates it).
4. (#404) Installing the wheel + the locally-built
   ``bucket-brigade-core`` wheel into a fresh venv yields a working
   ``bucket_brigade.make().reset()`` AND a successful import of
   ``bucket_brigade_core`` (the Rust ext is present, not just the
   pure-Python fallback).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUST_PKG_DIR = _REPO_ROOT / "bucket-brigade-core"


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def _have_cargo() -> bool:
    """``cargo`` on PATH is a proxy for "Rust toolchain installed"."""
    return shutil.which("cargo") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _have_uv(), reason="requires `uv` on PATH")
def test_uv_build_wheel_produces_installable_artifact(tmp_path: Path) -> None:
    """``uv build --wheel`` builds a wheel that contains the expected files."""
    # Build the wheel into an isolated dist directory so we don't
    # contaminate dist/ in the working copy.
    out_dir = tmp_path / "dist"
    out_dir.mkdir()

    result = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"uv build failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )

    wheels = list(out_dir.glob("bucket_brigade-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    wheel = wheels[0]
    assert wheel.stat().st_size > 10_000, "wheel is suspiciously small"

    # Inspect the wheel contents to confirm we ship the right tree.
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())

    # Core package marker file is present.
    assert "bucket_brigade/__init__.py" in names
    # Frozen-baselines plumbing is present.
    assert "bucket_brigade/baselines/release/__init__.py" in names
    assert "bucket_brigade/baselines/release/manifest.py" in names
    assert "bucket_brigade/baselines/release/paths.py" in names
    assert "bucket_brigade/baselines/release/hub.py" in names
    # In-wheel artifact dir exists (README.md is the placeholder we ship).
    assert "bucket_brigade/baselines/release/local/README.md" in names


def _build_bucket_brigade_wheel(dist_dir: Path) -> Path:
    """Run ``uv build --wheel`` for the top-level bucket-brigade package."""
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"uv build (bucket-brigade) failed:\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
    wheels = list(dist_dir.glob("bucket_brigade-*.whl"))
    if len(wheels) != 1:
        pytest.fail(f"expected one bucket_brigade wheel, got {wheels}")
    return wheels[0]


def _build_bucket_brigade_core_wheel(dist_dir: Path, python: str) -> Path:
    """Run ``maturin build`` for bucket-brigade-core.

    Returns the path to the produced wheel. The wheel lands directly in
    ``dist_dir``.

    ``python`` is the absolute path to the interpreter the wheel should be
    built against; we pass it through to ``maturin --interpreter`` so the
    cp3XX ABI tag in the wheel filename matches the venv we will later
    install into.
    """
    env = dict(os.environ)
    # Defensive: mirror the env vars build.sh sets so a CI runner with a
    # bare environment still builds successfully.
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
    env["RUSTC_WRAPPER"] = ""

    # Use the project's maturin (already in the dev venv) via ``uv run``
    # so we don't have to install maturin in a separate step.
    result = subprocess.run(
        [
            "uv",
            "run",
            "maturin",
            "build",
            "--release",
            "--features",
            "python",
            "--interpreter",
            python,
            "--out",
            str(dist_dir),
        ],
        cwd=str(_RUST_PKG_DIR),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"maturin build (bucket-brigade-core) failed:\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
    wheels = list(dist_dir.glob("bucket_brigade_core-*.whl"))
    if len(wheels) != 1:
        pytest.fail(f"expected one bucket_brigade_core wheel, got {wheels}")
    return wheels[0]


@pytest.mark.slow
@pytest.mark.skipif(not _have_uv(), reason="requires `uv` on PATH")
@pytest.mark.skipif(
    not _have_cargo(),
    reason="requires a Rust toolchain (cargo on PATH) to build bucket-brigade-core",
)
def test_wheel_imports_cleanly_in_fresh_venv(tmp_path: Path) -> None:
    """Install BOTH wheels into a fresh venv; verify env + Rust ext.

    Issue #404 acceptance criterion: ``pip install bucket-brigade``
    yields a working ``bucket_brigade.make().reset()`` AND has the Rust
    ext (``bucket_brigade_core``) available — without the historical
    second ``cd bucket-brigade-core && pip install .`` step.

    We simulate the "pip install bucket-brigade" experience by
    building both wheels locally and installing both into a fresh
    venv. The Rust wheel substitutes for the not-yet-published PyPI
    wheel that the cibuildwheel workflow will produce on tag push.
    """
    # 1. Fresh hermetic venv FIRST, so we can build the Rust wheel against
    #    its specific Python ABI. Ask for an interpreter that is in the
    #    supported range (``requires-python = ">=3.9,<3.14"``) — we pin to
    #    3.12 here because every Loom dev workstation and CI runner has it
    #    via ``uv``. Run uv with ``cwd=tmp_path`` so it doesn't pick up
    #    the workspace ``pyproject.toml`` and override our request.
    dist = tmp_path / "dist"
    dist.mkdir()
    venv = tmp_path / "venv"
    create = subprocess.run(
        ["uv", "venv", str(venv), "--python", "3.12"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode != 0:
        pytest.fail(f"uv venv failed:\n{create.stderr}")

    venv_python = venv / "bin" / "python"
    if not venv_python.exists():  # pragma: no cover - windows path
        venv_python = venv / "Scripts" / "python.exe"
    assert venv_python.exists(), f"no python in venv: {venv}"

    # 2. Build the bucket-brigade wheel (pure Python).
    bb_wheel = _build_bucket_brigade_wheel(dist)

    # 3. Build the bucket-brigade-core wheel (Rust + PyO3 via maturin)
    #    against the *fresh venv's* interpreter, so the cp3XX ABI tag in
    #    the wheel filename matches what we install into next.
    rust_wheel = _build_bucket_brigade_core_wheel(dist, str(venv_python))

    # 4. Install both wheels in a single command so uv sees the local
    #    Rust wheel as satisfying the bucket-brigade-core dep declared
    #    by bucket-brigade.
    install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            str(rust_wheel),
            str(bb_wheel),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode != 0:
        pytest.fail(
            f"uv pip install failed:\n"
            f"--- stdout ---\n{install.stdout}\n"
            f"--- stderr ---\n{install.stderr}\n"
        )

    # 5a. End-to-end env smoke test (PR #397 behaviour, still required).
    smoke = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import bucket_brigade; "
                "env = bucket_brigade.make('minimal_specialization-v1'); "
                "obs, info = env.reset(seed=0); "
                "print('OK', type(env).__name__, getattr(obs, 'shape', None))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if smoke.returncode != 0:
        pytest.fail(
            f"smoke test failed in fresh venv:\n"
            f"--- stdout ---\n{smoke.stdout}\n"
            f"--- stderr ---\n{smoke.stderr}\n"
        )
    assert smoke.stdout.startswith("OK ")

    # 5b. (#404) Rust ext is importable — confirms cibuildwheel-style
    #     install really did bundle the .so, not just rely on the
    #     pure-Python fallback.
    rust_check = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import bucket_brigade_core; "
                "from bucket_brigade_core import BucketBrigade, Scenario; "
                "print('RUST_OK', BucketBrigade.__name__, Scenario.__name__)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rust_check.returncode != 0:
        pytest.fail(
            f"Rust ext not importable from fresh venv (#404):\n"
            f"--- stdout ---\n{rust_check.stdout}\n"
            f"--- stderr ---\n{rust_check.stderr}\n"
        )
    assert rust_check.stdout.startswith("RUST_OK ")
