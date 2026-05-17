"""Anti-regression guard for the issue #199 baselines sampler.

PR #236 (issue #235) promoted the broadcast signal to a first-class
action dimension, widening ``MultiDiscrete([10, 2])`` to
``MultiDiscrete([10, 2, 2])``. The env does **not** validate input
action shape — a 2-dim action returns successfully but silently drops
the signal channel — so any baseline that keeps a 2-row sampler
silently measures a strictly different stochastic process than the
env actually exercises post-#236.

PR #244 fixed the sibling site in
``tests/test_env_health_diagnostics.py``. PR #248 fixed the sibling
site in ``pairwise_action_kl.py::softmax_packed``. This issue (#246)
fixes the third sibling site:
``experiments/p3_specialization/diagnostics/issue199_baselines.py``.

These assertions follow the same grep-the-source pattern PR #244
introduced. We deliberately do **not** import
``random_baseline.py`` (it transitively pulls in torch via the
training package and would break the no-RL CI install) and we
deliberately do **not** import ``issue199_baselines`` either — the
grep is sufficient and keeps this test runnable without RL extras.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAG_DIR = _REPO_ROOT / "experiments" / "p3_specialization" / "diagnostics"
_ISSUE199 = _DIAG_DIR / "issue199_baselines.py"
_CANONICAL = _DIAG_DIR / "random_baseline.py"


def test_issue199_baselines_sampler_is_3dim() -> None:
    """The issue199 random sampler must build a 3-column action array.

    We grep for three ``rng.integers(...)`` rows inside the
    ``np.stack([...], axis=-1)`` block of ``_run_random_episode``.
    Bare line count is sufficient because the helper is short and
    the only ``rng.integers`` calls in the file live inside that
    one stack.
    """
    src = _ISSUE199.read_text()
    n_rows = src.count("rng.integers(")
    assert n_rows >= 3, (
        f"issue199_baselines.py has only {n_rows} ``rng.integers(...)`` rows; "
        "the post-#236 MultiDiscrete([10, 2, 2]) sampler needs 3 (house, "
        "mode, signal). Without the third row the env silently drops the "
        "signal channel and this script measures a pre-#236 policy — the "
        "bug class PR #244 and PR #248 fixed in two sibling sites."
    )


def test_issue199_baselines_matches_canonical_action_dims() -> None:
    """Cross-check the canonical ACTION_DIMS constant has not drifted.

    Mirrors PR #244's anti-regression pattern in
    ``test_env_health_diagnostics.py`` lines 253-267. Greps the
    canonical source file rather than importing because
    ``random_baseline`` transitively imports torch via the training
    package, and this test must remain runnable in the no-RL CI
    install.
    """
    rb_src = _CANONICAL.read_text()
    assert "ACTION_DIMS = [10, 2, 2]" in rb_src, (
        "random_baseline.ACTION_DIMS no longer equals [10, 2, 2]. "
        "Update issue199_baselines.py's sampler AND the grep above to "
        "match the new layout — otherwise issue199_baselines silently "
        "measures the wrong policy (the bug class PR #244 fixed)."
    )
