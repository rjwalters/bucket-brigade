"""H1 diagnostic: inspect per-step ``rollout.rewards`` distribution (#190).

Loads trained policies from a P3 cell, runs one rollout via
``trainer.collect_rollout(2048)``, and dumps per-agent reward statistics
to confirm/rule out reward-signal degeneracy (hypothesis H1 from #190 /
#183).

Cell source
-----------
Default cell: ``/tmp/h1_cell`` — rsynced from ``$COMPUTE_HOST_PRIMARY``
(``robbs-mac-studio``) at:

    ~/GitHub/bucket-brigade/experiments/p3_specialization/runs/
        p3_183_phase3/L1_norm/default/lambda_0e0/seed_42/

That cell ran 500 iterations (scenario=default, lambda_red=0.0, seed=42,
normalize_returns=true). It is the canonical Phase-3-stuck cell cited in
the #190 issue body.

If ``/tmp/h1_cell`` is missing, fall back to a freshly trained 50-iter
cell with the same hyperparameters; the H1 question is about signal
degeneracy, which is visible at any training stage. Override the cell
path with ``--cell <path>``.

Methodology (per curator enhancement on #190)
---------------------------------------------
``rollout.rewards`` is ``Dict[int, torch.Tensor]`` — one ``[T]`` tensor
per agent — and must be stacked into an ``ndarray[N, T]`` before
distributional analysis. For each agent we report:

- mean, std, CV (= std / |mean|)
- p10 / p50 / p90 of per-step reward
- aggregate text histogram across all (N, T) samples
- analytic decomposition of the action-controllable work/rest term
  (constants from the ``default`` scenario: ``cost_to_work_one_night=0.5``,
  rest_reward=+0.5) and the residual (team + ownership + sparking)
- action–reward R² via a 20-class regressor on the packed action
  ``a[:, 0] * 2 + a[:, 1]`` (10 houses × {rest, work})
- a side-channel R² on the binary work/rest dim alone, for sanity

Interpretive rubric (from issue #190 ACs):

- CV < 0.05 AND R² < 0.01  → H1 confirmed (degenerate signal).
- CV low but R² >= 0.05    → H1 ruled out; PPO is failing to extract
  the signal (points at H2/H3/H4).
- CV high AND R² >= 0.05   → H1 ruled out; reward is informative.

Usage
-----
    uv run python experiments/p3_specialization/diagnostics/inspect_rollout_rewards.py
    uv run python ... --cell /tmp/h1_cell --no-baseline
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import (
    JointPPOTrainer,
    flatten_dict_obs,
)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def per_agent_stats(r: np.ndarray) -> dict:
    """Distributional stats for a single agent's per-step reward series."""
    mean = float(r.mean())
    std = float(r.std())
    cv = std / (abs(mean) + 1e-9)
    p10, p50, p90 = (float(x) for x in np.percentile(r, [10, 50, 90]))
    return {"mean": mean, "std": std, "cv": cv, "p10": p10, "p50": p50, "p90": p90}


def conditional_mean_r2(r: np.ndarray, labels: np.ndarray) -> float:
    """R² of the optimal class-mean regressor of ``r`` on ``labels``.

    Equivalent to 1 − SS_within / SS_total — i.e. the fraction of
    per-step reward variance explained by the action label.
    Returns NaN if ``r`` has zero variance.
    """
    ss_total = float(((r - r.mean()) ** 2).sum())
    if ss_total <= 1e-12:
        return float("nan")
    pred = np.zeros_like(r, dtype=np.float64)
    for lab in np.unique(labels):
        mask = labels == lab
        pred[mask] = r[mask].mean()
    ss_res = float(((r - pred) ** 2).sum())
    return 1.0 - ss_res / ss_total


def text_histogram(values: np.ndarray, n_bins: int = 15, width: int = 60) -> str:
    """ASCII histogram of a 1-D array."""
    hist, edges = np.histogram(values, bins=n_bins)
    peak = max(hist.max(), 1)
    lines = []
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(width * h / peak)
        lines.append(f"  [{lo:8.3f}, {hi:8.3f}) {h:6d} {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------


def run_one_rollout(
    cell_dir: Path,
    *,
    load_policies: bool,
    rollout_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, "BucketBrigadeEnv"]:
    """Load (or randomly init) policies from ``cell_dir`` and run one rollout.

    Returns ``(R, A, cfg, scenario)`` where:

    - ``R``: ndarray ``[N, T]`` of per-step rewards (stacked from the dict).
    - ``A``: ndarray ``[N, T, 2]`` of per-step actions.
    - ``cfg``: dict from ``config.json``.
    - ``scenario``: the env scenario object (for analytic decomposition).
    """
    cfg = json.loads((cell_dir / "config.json").read_text())
    scenario = get_scenario_by_name(cfg["scenario"], num_agents=cfg["num_agents"])

    def env_fn():
        return BucketBrigadeEnv(scenario=scenario)

    probe = env_fn()
    obs_dim = flatten_dict_obs(probe.reset(seed=cfg["seed"])).shape[0]

    trainer = JointPPOTrainer(
        env_fn=env_fn,
        num_agents=cfg["num_agents"],
        obs_dim=obs_dim,
        action_dims=cfg["action_dims"],
        hidden_size=cfg["hidden_size"],
        lr=cfg["lr"],
        ppo_epochs=cfg["ppo_epochs"],
        minibatch_size=cfg["minibatch_size"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        normalize_returns=cfg.get("normalize_returns", False),
        device=cfg.get("device", "cpu"),
        seed=cfg["seed"],
    )

    if load_policies:
        for i in range(cfg["num_agents"]):
            sd = torch.load(
                cell_dir / f"policies/agent_{i}.pt",
                map_location=cfg.get("device", "cpu"),
                weights_only=True,
            )
            trainer.policies[i].load_state_dict(sd)

    steps = rollout_steps if rollout_steps is not None else cfg["rollout_steps"]
    rollout = trainer.collect_rollout(steps)

    # Sanity-check the curator's claim about dtype/shape before we proceed.
    assert isinstance(rollout.rewards, dict), (
        f"rollout.rewards is {type(rollout.rewards)}, expected dict — STOP"
    )
    sample = rollout.rewards[0]
    assert isinstance(sample, torch.Tensor) and sample.ndim == 1, (
        f"rollout.rewards[0] is {type(sample)} with shape {getattr(sample, 'shape', None)}; "
        "expected torch.Tensor[T]"
    )

    n = cfg["num_agents"]
    R = torch.stack([rollout.rewards[i] for i in range(n)]).cpu().numpy()
    A = torch.stack([rollout.actions[i] for i in range(n)]).cpu().numpy()
    return R, A, cfg, scenario


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(
    label: str,
    R: np.ndarray,
    A: np.ndarray,
    scenario,
) -> dict:
    """Print + return the per-agent verdict block for one rollout."""
    n = R.shape[0]
    work_cost = float(scenario.cost_to_work_one_night)
    rest_reward = 0.5  # hard-coded in BucketBrigadeEnv._compute_rewards
    print(
        f"\n{'=' * 72}\n{label}\nR.shape = {R.shape}, A.shape = {A.shape}\n{'=' * 72}"
    )

    summary = {}
    for i in range(n):
        r = R[i]
        a = A[i]  # [T, 2]
        stats = per_agent_stats(r)

        # Action-reward R² on the packed 20-class action (10 houses x 2 modes).
        packed = a[:, 0] * 2 + a[:, 1]
        r2_packed = conditional_mean_r2(r, packed)
        # Side-channel: binary work/rest only.
        r2_work = conditional_mean_r2(r, a[:, 1])

        # Analytic decomposition of the work/rest term:
        # +0.5 on rest steps, -cost_to_work_one_night on work steps.
        work_mask = a[:, 1] == 1  # WORK == 1
        wr_term = np.where(work_mask, -work_cost, rest_reward).astype(np.float32)
        residual = r - wr_term  # team + ownership (+ stochastic env)

        wr_mean = float(wr_term.mean())
        wr_std = float(wr_term.std())
        res_mean = float(residual.mean())
        res_std = float(residual.std())

        # Verdict per H1 rubric (#190 ACs)
        h1_cv = stats["cv"] < 0.05
        h1_r2 = r2_packed < 0.01 if not np.isnan(r2_packed) else False
        if h1_cv and h1_r2:
            verdict = "H1 CONFIRMED (degenerate signal)"
        elif (not h1_cv) and (not h1_r2):
            verdict = "H1 RULED OUT (signal varies and is action-coupled)"
        else:
            verdict = "H1 AMBIGUOUS (mixed signal — see decomposition)"

        print(
            f"\nagent_{i}:"
            f"\n  mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
            f"CV={stats['cv']:.4f}"
            f"\n  p10/p50/p90 = {stats['p10']:.3f} / {stats['p50']:.3f} / {stats['p90']:.3f}"
            f"\n  work_rate = {work_mask.mean():.3f}  "
            f"wr_term(mean,std) = ({wr_mean:.3f}, {wr_std:.3f})  "
            f"residual(mean,std) = ({res_mean:.3f}, {res_std:.3f})"
            f"\n  R²(packed_20cls) = {r2_packed:.6f}  "
            f"R²(work/rest only) = {r2_work:.6f}"
            f"\n  --> {verdict}"
        )

        summary[f"agent_{i}"] = {
            **stats,
            "work_rate": float(work_mask.mean()),
            "wr_term_mean": wr_mean,
            "wr_term_std": wr_std,
            "residual_mean": res_mean,
            "residual_std": res_std,
            "r2_packed": float(r2_packed),
            "r2_work": float(r2_work),
            "verdict": verdict,
        }

    print("\nHistogram of all per-step rewards (flattened across agents and steps):")
    print(text_histogram(R.flatten()))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        type=Path,
        default=Path(tempfile.gettempdir()) / "h1_cell",
        help=(
            "Trained-cell directory containing config.json + policies/. "
            "Defaults to <system-tempdir>/h1_cell (the conventional rsync "
            "target documented in the script docstring)."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the random-init comparison rollout.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path to save the numeric summary.",
    )
    args = parser.parse_args()

    if not (args.cell / "config.json").exists():
        raise SystemExit(
            f"Cell not found at {args.cell}. See the script docstring for "
            "rsync instructions, or train a fresh 50-iter cell."
        )

    # 1) Trained-policy rollout.
    R, A, cfg, scenario = run_one_rollout(args.cell, load_policies=True)
    trained_summary = report(
        f"TRAINED CELL: {args.cell}  "
        f"(scenario={cfg['scenario']}, lambda_red={cfg['lambda_red']}, "
        f"seed={cfg['seed']}, num_iterations={cfg['num_iterations']})",
        R,
        A,
        scenario,
    )

    # 2) Random-init baseline rollout (same env, same config).
    baseline_summary = None
    if not args.no_baseline:
        Rb, Ab, _, _ = run_one_rollout(args.cell, load_policies=False)
        baseline_summary = report(
            "RANDOM-INIT BASELINE (same config, no loaded policies)",
            Rb,
            Ab,
            scenario,
        )

    # 3) Optional: dump numerics to JSON for the report.
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "cell": str(args.cell),
                    "cfg": cfg,
                    "trained": trained_summary,
                    "random_baseline": baseline_summary,
                },
                indent=2,
            )
        )
        print(f"\nWrote summary JSON to {args.out}")


if __name__ == "__main__":
    main()
