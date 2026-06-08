"""Pip-wheel build smoke test (issue #373).

Marked ``slow`` because it shells out to ``uv build`` and builds an
actual wheel; not part of the fast-test gate but is run by the
``-m "not slow"`` carve-out and by anyone validating a release.

Validates:

1. ``uv build --wheel`` succeeds from the repo root.
2. The produced ``.whl`` is non-empty and contains
   ``bucket_brigade/__init__.py``.
3. The wheel contains the ``baselines/release/local/`` placeholder
   (so the in-wheel artifact dir exists once #371 populates it).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _have_uv() -> bool:
    return shutil.which("uv") is not None


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


@pytest.mark.slow
@pytest.mark.skipif(not _have_uv(), reason="requires `uv` on PATH")
def test_wheel_imports_cleanly_in_fresh_venv(tmp_path: Path) -> None:
    """Install the built wheel into a fresh venv and call ``make()``.

    This is the smoke test from the issue's acceptance criteria:
    ``pip install bucket-brigade && python -c "import bucket_brigade;
    bucket_brigade.make('minimal_specialization-v1').reset()"``.
    """
    # Build the wheel.
    dist = tmp_path / "dist"
    dist.mkdir()
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.fail(f"uv build failed:\n{build.stderr}")

    wheels = list(dist.glob("bucket_brigade-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    # Create a fresh venv via uv. uv venv is hermetic.
    venv = tmp_path / "venv"
    create = subprocess.run(
        ["uv", "venv", str(venv), "--python", sys.executable],
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

    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheel)],
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
