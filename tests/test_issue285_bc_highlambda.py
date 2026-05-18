"""Tests for issue #285 — BC-init + high-λ PPO combined warm-start probe.

This issue composes #270's BC-init flag (``--bc-init-checkpoint-dir``) with
#282's high-λ flag (``--gae-lambda``). Both flags were already wired by
PR #295 (issue #282) and PR #278 (issue #270); this test module verifies
the **composition** and the analyzer + sweep driver scaffolding added
for the 2×2 verdict matrix.

Tests:

1. λ=0.95 + BC-init produces bit-identical trajectories vs. BC-init-only
   (no ``--gae-lambda`` flag) — guards the default-preserving behavior of
   the #282 plumbing when combined with the #270 BC-init path.
2. λ=1.0 + BC-init runs end-to-end without producing NaN advantages or
   losses (numerical stability check at the pure-MC edge of the grid).
3. ``compute_gae`` advantage variance increases monotonically with λ on
   a contrived non-zero-reward / non-terminal trajectory — sanity check
   for the bias-variance tradeoff (higher λ → higher variance).
4. ``analyze_285`` per-λ aggregator round-trips a synthetic metrics layout
   into the expected verdict-ladder JSON structure.

All tests are CPU-only and finish in seconds; PPO is run with
``num_iterations=1`` / ``rollout_steps=64`` to stay within the local
smoke-test budget per ``CLAUDE.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "p3_specialization"))

import bc_init  # type: ignore[import-not-found]  # noqa: E402
from bucket_brigade.training.networks import compute_gae  # noqa: E402


@pytest.fixture(scope="module")
def tiny_bc_checkpoint(tmp_path_factory):
    """Produce a tiny BC checkpoint dir for downstream tests.

    Uses 32 demo pairs/agent and 1 epoch — just enough to populate the
    state dict for the warm-start load path. The actual BC fit quality
    is irrelevant; we only need the load path to engage.
    """
    out = tmp_path_factory.mktemp("issue285_bc_smoke")
    num_agents = 4

    obs_per_agent, act_per_agent = bc_init.gather_demos(
        scenario_name="minimal_specialization",
        num_agents=num_agents,
        num_pairs_per_agent=32,
        seed=0,
    )
    obs_dim = obs_per_agent[0].shape[1]
    for i in range(num_agents):
        p, _ = bc_init.bc_fit_one_agent(
            obs=obs_per_agent[i],
            acts=act_per_agent[i],
            obs_dim=obs_dim,
            action_dims=[10, 2, 2],
            hidden_size=64,
            epochs=1,
            batch_size=32,
            lr=1e-3,
            device="cpu",
            seed=i,
        )
        torch.save(p.state_dict(), out / f"agent_{i}.pt")
    return out


def _read_metrics(cell_dir: Path) -> List[dict]:
    return json.loads((cell_dir / "metrics.json").read_text())


def _trajectory(metrics: List[dict]) -> np.ndarray:
    return np.asarray(
        [row["mean_step_reward_team"] for row in metrics], dtype=np.float64
    )


def _train_one_smoke(
    *,
    out_dir: Path,
    bc_dir: Path,
    gae_lambda: float | None,
    seed: int = 123,
):
    """Run a tiny 1-iter PPO training with optional ``gae_lambda`` override."""
    import train  # type: ignore[import-not-found]

    kwargs = dict(
        scenario="minimal_specialization",
        lambda_red=0.0,
        seed=seed,
        num_iterations=1,
        rollout_steps=64,
        num_agents=4,
        bc_init_checkpoint_dir=str(bc_dir),
    )
    if gae_lambda is not None:
        kwargs["gae_lambda"] = gae_lambda
    cfg = train.CellConfig(**kwargs)
    train.train_one_cell(cfg, out_dir)
    return out_dir


def test_lambda_default_matches_bc_init_only(tiny_bc_checkpoint, tmp_path):
    """λ=0.95 (default) with BC-init must match BC-init-only without --gae-lambda.

    Both code paths feed ``CellConfig.gae_lambda`` into ``JointPPOTrainer``;
    when the user passes ``--gae-lambda 0.95`` (or omits the flag) the
    constructor argument is identical, so trajectories must be bit-identical.
    This guards the no-op default-preserving behavior of the #282 plumbing
    when composed with the #270 BC-init load path (the issue #285 composition
    we're validating).
    """
    out_default = tmp_path / "default"
    out_explicit = tmp_path / "explicit_0_95"
    _train_one_smoke(
        out_dir=out_default, bc_dir=tiny_bc_checkpoint, gae_lambda=None, seed=123
    )
    _train_one_smoke(
        out_dir=out_explicit, bc_dir=tiny_bc_checkpoint, gae_lambda=0.95, seed=123
    )

    m_default = _trajectory(_read_metrics(out_default))
    m_explicit = _trajectory(_read_metrics(out_explicit))
    np.testing.assert_array_equal(
        m_default,
        m_explicit,
        err_msg="λ=0.95 explicit must be bit-identical to default (no flag).",
    )


def test_lambda_one_no_nan(tiny_bc_checkpoint, tmp_path):
    """λ=1.0 (pure MC) with BC-init must complete one iter without NaN metrics.

    The pure-Monte-Carlo edge of the λ grid is the highest-variance cell in
    the issue #285 sweep and the most likely to produce numerical issues. We
    don't assert anything about reward magnitude — just that the metrics
    log contains finite floats end-to-end.
    """
    out = tmp_path / "lambda_1_0"
    _train_one_smoke(
        out_dir=out, bc_dir=tiny_bc_checkpoint, gae_lambda=1.0, seed=123
    )
    metrics = _read_metrics(out)
    assert len(metrics) == 1, "Expected exactly one iteration of metrics."
    for key, value in metrics[0].items():
        if isinstance(value, float):
            assert np.isfinite(value), (
                f"Non-finite metric at λ=1.0: {key} = {value}"
            )


def test_gae_variance_increases_with_lambda():
    """Higher GAE λ → higher advantage variance on a contrived trajectory.

    Constructs a non-terminal, non-zero-reward sequence where the GAE
    geometric sum genuinely depends on λ (a constant-zero rewards/values
    sequence would have identically zero advantages at every λ). The
    bias-variance tradeoff in Schulman et al. (2016) says that as λ → 1
    we approach the Monte Carlo return, which has the highest variance
    among GAE estimators (and zero bias). We check this for the issue
    #285 sweep grid — which only contains λ ∈ {0.95, 0.99, 0.999, 1.0},
    i.e., the high-λ regime where the variance ordering is well-defined.
    (At very low λ, e.g. comparing λ=0 to λ=0.5, the ordering can flip
    on a noisy seed because the single-step TD error happens to dominate
    the geometric sum; we deliberately do not span the full grid here.)
    """
    rng = np.random.default_rng(0)
    T = 256
    rewards = rng.standard_normal(T).astype(np.float32).tolist()
    values = rng.standard_normal(T).astype(np.float32).tolist()
    dones = [False] * T  # non-terminating — maximize the λ effect

    sweep_lambdas = (0.95, 0.99, 0.999, 1.0)
    variances: List[float] = []
    for lam in sweep_lambdas:
        adv = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=lam)
        variances.append(float(np.var(adv)))

    # Variance must rise monotonically across the issue #285 λ grid.
    for i in range(1, len(variances)):
        assert variances[i] > variances[i - 1], (
            f"GAE variance should rise with λ across {sweep_lambdas}; "
            f"saw {variances} (violation at index {i})"
        )


def test_analyze_285_roundtrip(tmp_path):
    """analyze_285 aggregates synthetic metrics into the expected layout.

    Builds a fake 4 λ × 3 seed metrics tree and checks that
    ``analyze_285.main()`` writes ``analysis.json`` with the per-λ verdict
    ladder and the (skipped) reference cross-check stanza.
    """
    import importlib

    analyze_285 = importlib.import_module("analyze_285")

    runs_root = tmp_path / "runs"
    # Use the same λ tag convention as the sweep driver.
    for lam in (0.95, 0.99, 0.999, 1.0):
        tag = str(lam).replace(".", "_")
        for s in (42, 43, 44):
            cell = runs_root / f"lambda_{tag}" / f"seed_{s}"
            cell.mkdir(parents=True, exist_ok=True)
            # Synthesize 10 iters of per-step team rewards. Bias higher-λ
            # cells slightly upward so the "best λ" picker exercises the
            # `max` branch and produces a non-NaN best.
            base = -50.0 + 5.0 * (lam - 0.95)
            traj = [
                {"mean_step_reward_team": base + 0.1 * t}
                for t in range(10)
            ]
            (cell / "metrics.json").write_text(json.dumps(traj))

    output_dir = tmp_path / "diagnostics"
    # Invoke the CLI via sys.argv (matches how it's run in production).
    argv_backup = sys.argv
    try:
        sys.argv = [
            "analyze_285",
            "--runs-root",
            str(runs_root),
            "--seeds",
            "42",
            "43",
            "44",
            "--output-dir",
            str(output_dir),
        ]
        analyze_285.main()
    finally:
        sys.argv = argv_backup

    result = json.loads((output_dir / "analysis.json").read_text())
    assert result["issue"] == 285
    assert "best_lambda" in result
    assert result["best_lambda"] is not None
    # All four λ cells should have been aggregated.
    assert set(result["per_lambda_verdicts"].keys()) == {
        "0.95",
        "0.99",
        "0.999",
        "1.0",
    }
    for lam_key, entry in result["per_lambda_verdicts"].items():
        assert entry["n_seeds"] == 3, f"Expected 3 seeds for λ={lam_key}"
        assert "verdict" in entry
        assert entry["verdict"] in {
            "basin_trap",
            "anti_attractor",
            "partial",
            "bc_did_not_take",
            "no_data",
        }
    # No --reference-270-root provided: cross-check stanza must be skipped.
    assert result["reference_270_cross_check"]["status"] == "skipped"
    # Markdown sidecar exists.
    assert (output_dir / "verdict.md").exists()
