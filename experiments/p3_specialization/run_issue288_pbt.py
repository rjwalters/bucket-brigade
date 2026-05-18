"""Issue #288 — Population-Based Training (PBT) orchestrator.

Thin wrapper (Option A per curator) over ``experiments/p3_specialization/train.py``.
Spawns ``population_size`` independent PPO lineages on ``minimal_specialization``,
runs ``generations`` * ``iters_per_gen`` PPO iterations per lineage, and between
generations applies Jaderberg-style truncation selection + weight/hyperparameter
mutation:

- **Exploit**: bottom 25% (rounded down) of lineages, ranked by trailing-5 mean
  step team-reward over the most recent generation, are replaced by perturbed
  copies of randomly-chosen top-25% donors.
- **Explore (weights)**: donor checkpoints are loaded, additive Gaussian noise
  ``σ = weight_noise * std(layer)`` is added to every tensor, and the perturbed
  state dicts are saved to a per-lineage ``perturbed_init/`` directory which
  becomes the next generation's ``--bc-init-checkpoint-dir``.
- **Explore (hyperparams)**: ``lr`` is multiplied by uniform({0.8, 1.25}) with
  probability 0.5; ``entropy_coef`` is multiplied by uniform({0.5, 2.0}) with
  probability 0.5. Surviving (top 75%) lineages keep their hyperparameters and
  warm-start from their own previous generation's policies.

Layout on disk::

    <output_dir>/
        seed_<PBT_SEED>/
            gen_0/
                lineage_0/{metrics.json,config.json,policies/agent_*.pt,train.log}
                ...
                lineage_<P-1>/...
                ranking.json          # post-generation ranking + exploit plan
            gen_1/
                lineage_0/
                    perturbed_init/agent_*.pt   # (only if this lineage was replaced)
                    metrics.json
                    ...
                ...
            lineage_state.json        # final per-lineage hyperparams + history

This script is **safe to run locally only as a smoke test** (e.g.,
``--population-size 4 --generations 2 --iters-per-gen 5``). The full
``--population-size 16 --generations 6 --iters-per-gen 50`` run is CPU-heavy
and MUST go on ``COMPUTE_HOST_PRIMARY`` per ``CLAUDE.md`` guidelines; see
``run_issue288_pbt.sh``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess  # nosec B404 (orchestrator spawns train.py with fixed argv)
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "p3_specialization" / "train.py"

# Trailing-N for selection criterion. Mirrors analyze_270.py's TRAILING_N
# so the orchestrator's ranking signal matches the verdict-classifier's input.
TRAILING_N = 5

# Jaderberg-style multiplicative perturbation factors. Applied with prob
# ``mutation_prob`` independently per knob per replaced lineage.
LR_PERTURB_CHOICES: Sequence[float] = (0.8, 1.25)
ENTROPY_PERTURB_CHOICES: Sequence[float] = (0.5, 2.0)
MUTATION_PROB = 0.5


@dataclass
class LineageState:
    """Mutable per-lineage state carried across generations."""

    lineage_id: int
    seed: int
    lr: float
    entropy_coef: float
    # If non-None, this is the directory containing agent_{i}.pt files to load
    # into trainer.policies before PPO starts. After the first generation,
    # surviving lineages point at their own previous gen's ``policies/`` dir;
    # exploited lineages point at a ``perturbed_init/`` dir written by
    # ``_apply_exploit_explore``.
    init_checkpoint_dir: Optional[str] = None
    # Per-generation trailing-5 mean step team reward (for diagnostics + final
    # verdict). Indexed by generation number.
    trailing5_team_history: List[float] = field(default_factory=list)
    # Provenance: which donor lineage we were spawned from at each generation
    # (None = survived, no replacement).
    donor_history: List[Optional[int]] = field(default_factory=list)


def _trailing5_team_reward(metrics_path: Path) -> Optional[float]:
    """Read metrics.json, return mean of last TRAILING_N mean_step_reward_team.

    Returns None when the file is missing or empty (subprocess crashed). Caller
    treats None as the worst possible score so crashed lineages get replaced.
    """
    if not metrics_path.exists():
        return None
    try:
        rows = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None
    if not rows:
        return None
    tail = rows[-TRAILING_N:]
    return float(sum(r["mean_step_reward_team"] for r in tail) / max(1, len(tail)))


def _run_lineage_cell(
    state: LineageState,
    scenario: str,
    lambda_red: float,
    num_agents: int,
    rollout_steps: int,
    iters: int,
    output_dir: Path,
) -> int:
    """Spawn train.py for one lineage's generation. Returns exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--scenario",
        scenario,
        "--lambda-red",
        str(lambda_red),
        "--seed",
        str(state.seed),
        "--num-iterations",
        str(iters),
        "--rollout-steps",
        str(rollout_steps),
        "--num-agents",
        str(num_agents),
        "--lr",
        str(state.lr),
        "--entropy-coef",
        str(state.entropy_coef),
        "--output-dir",
        str(output_dir),
    ]
    if state.init_checkpoint_dir is not None:
        cmd.extend(["--bc-init-checkpoint-dir", state.init_checkpoint_dir])

    log_path = output_dir / "train.log"
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)  # nosec B603 (cmd is list, no shell)
    return proc.returncode


def _perturb_checkpoint_dir(
    src_dir: Path, dst_dir: Path, sigma: float, rng: random.Random
) -> None:
    """Copy donor checkpoints to ``dst_dir`` and add Gaussian noise in-place.

    Noise scale is ``sigma * std(tensor)`` per-tensor (per-layer). The "layer"
    granularity is whatever the donor's ``state_dict`` keys carve up — for the
    MLP policies in this stack that's one entry per ``weight``/``bias``. Zero-std
    tensors (e.g., zero-init biases) get plain ``N(0, sigma)`` noise so the
    perturbation isn't silently a no-op.

    Pickle compatibility: torch saves+loads the perturbed tensors via the same
    code path the trainer uses for ``--bc-init-checkpoint-dir``, so the only
    failure mode is shape/dtype drift (which load_state_dict catches strictly).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(src_dir.glob("agent_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no agent_*.pt under {src_dir}")
    for src_ckpt in ckpts:
        sd = torch.load(src_ckpt, map_location="cpu", weights_only=True)
        for key, tensor in sd.items():
            if not torch.is_floating_point(tensor):
                continue
            std = float(tensor.std().item()) if tensor.numel() > 1 else 0.0
            # rng -> torch generator: a single seed per tensor so the noise is
            # reproducible from the orchestrator's RNG state.
            g = torch.Generator()
            g.manual_seed(rng.randint(0, 2**31 - 1))
            scale = sigma * std if std > 0.0 else sigma
            noise = torch.randn(tensor.shape, generator=g) * scale
            sd[key] = tensor + noise.to(tensor.dtype)
        torch.save(sd, dst_dir / src_ckpt.name)


def _apply_exploit_explore(
    lineages: List[LineageState],
    gen_dir: Path,
    next_gen_dir: Path,
    weight_noise: float,
    rng: random.Random,
    truncation_frac: float = 0.25,
) -> Dict:
    """Rank lineages by their just-completed generation, exploit/explore the
    bottom ``truncation_frac``.

    For each lineage:

    - If it survived (ranked above the truncation cutoff), update its
      ``init_checkpoint_dir`` to point at this generation's ``policies/`` dir.
      Hyperparameters stay as-is.
    - If it was exploited (below the cutoff), pick a top-25% donor uniformly
      at random, write a perturbed copy of the donor's checkpoints to
      ``next_gen_dir/lineage_<id>/perturbed_init/``, and mutate ``lr`` and
      ``entropy_coef`` per Jaderberg perturbation factors with probability
      ``MUTATION_PROB`` each.

    Returns the ranking dict for the on-disk ``ranking.json`` audit trail.
    """
    pop_size = len(lineages)
    if pop_size == 0:
        return {"population_size": 0, "ranking": [], "replacements": []}

    # Score every lineage from its just-finished generation cell.
    scored: List[Dict] = []
    for lineage in lineages:
        cell_dir = gen_dir / f"lineage_{lineage.lineage_id}"
        score = _trailing5_team_reward(cell_dir / "metrics.json")
        # Track the trailing-5 score history regardless of replacement.
        lineage.trailing5_team_history.append(float("-inf") if score is None else score)
        scored.append(
            {
                "lineage_id": lineage.lineage_id,
                "trailing5_team_reward": score,
                "missing_metrics": score is None,
            }
        )

    # Sort descending; None scores go to the bottom (replaced).
    def sort_key(entry: Dict) -> float:
        s = entry["trailing5_team_reward"]
        return float("-inf") if s is None else s

    ranked = sorted(scored, key=sort_key, reverse=True)
    n_bottom = max(1, math.floor(truncation_frac * pop_size))
    # Top-25%: at least 1, symmetric with bottom.
    n_top = max(1, math.floor(truncation_frac * pop_size))
    top_ids = [entry["lineage_id"] for entry in ranked[:n_top]]
    bottom_ids = {entry["lineage_id"] for entry in ranked[-n_bottom:]}

    replacements: List[Dict] = []
    for lineage in lineages:
        gen_cell = gen_dir / f"lineage_{lineage.lineage_id}"
        survived_policies = gen_cell / "policies"
        if lineage.lineage_id in bottom_ids:
            donor_id = rng.choice(top_ids)
            donor_policies = gen_dir / f"lineage_{donor_id}" / "policies"
            perturbed_dir = (
                next_gen_dir / f"lineage_{lineage.lineage_id}" / "perturbed_init"
            )
            _perturb_checkpoint_dir(donor_policies, perturbed_dir, weight_noise, rng)
            lineage.init_checkpoint_dir = str(perturbed_dir)

            # Inherit donor's hyperparams as the basis for mutation.
            donor_state = next(s for s in lineages if s.lineage_id == donor_id)
            new_lr = donor_state.lr
            new_entropy = donor_state.entropy_coef
            lr_factor: Optional[float] = None
            ent_factor: Optional[float] = None
            if rng.random() < MUTATION_PROB:
                lr_factor = rng.choice(LR_PERTURB_CHOICES)
                new_lr = donor_state.lr * lr_factor
            if rng.random() < MUTATION_PROB:
                ent_factor = rng.choice(ENTROPY_PERTURB_CHOICES)
                new_entropy = donor_state.entropy_coef * ent_factor
            lineage.lr = new_lr
            lineage.entropy_coef = new_entropy
            lineage.donor_history.append(donor_id)
            replacements.append(
                {
                    "lineage_id": lineage.lineage_id,
                    "donor_lineage_id": donor_id,
                    "lr_perturb_factor": lr_factor,
                    "entropy_perturb_factor": ent_factor,
                    "new_lr": new_lr,
                    "new_entropy_coef": new_entropy,
                }
            )
        else:
            # Surviving lineage: warm-start the next generation from its own
            # final policies. Hyperparameters unchanged.
            lineage.init_checkpoint_dir = str(survived_policies)
            lineage.donor_history.append(None)

    return {
        "population_size": pop_size,
        "n_top": n_top,
        "n_bottom": n_bottom,
        "ranking": ranked,
        "replacements": replacements,
    }


def run_pbt(
    *,
    output_root: Path,
    pbt_seed: int,
    population_size: int,
    generations: int,
    iters_per_gen: int,
    scenario: str,
    lambda_red: float,
    num_agents: int,
    rollout_steps: int,
    initial_lr: float,
    initial_entropy_coef: float,
    weight_noise: float,
    truncation_frac: float = 0.25,
) -> Dict:
    """Run one PBT trial (one ``pbt_seed``). Returns a summary dict."""
    seed_root = output_root / f"seed_{pbt_seed}"
    seed_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(pbt_seed)  # nosec B311 (PBT mutation seeding, not cryptographic)

    # Distinct per-lineage PPO seeds. Using ``pbt_seed * 1000 + i`` keeps the
    # numbering human-readable in the per-cell config.json.
    lineages: List[LineageState] = [
        LineageState(
            lineage_id=i,
            seed=pbt_seed * 1000 + i,
            lr=initial_lr,
            entropy_coef=initial_entropy_coef,
        )
        for i in range(population_size)
    ]

    for gen in range(generations):
        gen_dir = seed_root / f"gen_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Sequential lineage runs. Parallelism comes from the launcher script
        # (xargs -P N) on the remote host; running parallel here would double-
        # parallelize and starve CPU on a single workstation.
        for lineage in lineages:
            cell_dir = gen_dir / f"lineage_{lineage.lineage_id}"
            rc = _run_lineage_cell(
                state=lineage,
                scenario=scenario,
                lambda_red=lambda_red,
                num_agents=num_agents,
                rollout_steps=rollout_steps,
                iters=iters_per_gen,
                output_dir=cell_dir,
            )
            if rc != 0:
                # Log but don't abort: the missing-metrics sentinel sinks the
                # lineage to the bottom of the ranking so it gets replaced.
                print(
                    f"  [WARN] lineage {lineage.lineage_id} gen {gen} "
                    f"exited with code {rc} — will be exploited",
                    flush=True,
                )

        # Exploit + explore unless this was the last generation (no need to
        # mutate after the final cell — surviving best is the verdict).
        next_gen_dir = seed_root / f"gen_{gen + 1}"
        if gen < generations - 1:
            ranking = _apply_exploit_explore(
                lineages,
                gen_dir,
                next_gen_dir,
                weight_noise=weight_noise,
                rng=rng,
                truncation_frac=truncation_frac,
            )
        else:
            # Final generation: no mutation applied. Score every lineage from
            # its just-finished cell and append to history so analyzers see a
            # uniformly-shaped trailing5 trajectory across all generations.
            for lineage in lineages:
                cell_dir = gen_dir / f"lineage_{lineage.lineage_id}"
                score = _trailing5_team_reward(cell_dir / "metrics.json")
                lineage.trailing5_team_history.append(
                    float("-inf") if score is None else score
                )
            ranking = {
                "population_size": len(lineages),
                "ranking": [
                    {
                        "lineage_id": lin.lineage_id,
                        "trailing5_team_reward": lin.trailing5_team_history[-1],
                    }
                    for lin in sorted(
                        lineages,
                        key=lambda x: x.trailing5_team_history[-1],
                        reverse=True,
                    )
                ],
                "replacements": [],
            }

        (gen_dir / "ranking.json").write_text(json.dumps(ranking, indent=2))
        print(
            f"== seed {pbt_seed} gen {gen}/{generations - 1}: "
            f"best trailing5 = {ranking['ranking'][0].get('trailing5_team_reward')}",
            flush=True,
        )

    # Persist final per-lineage state so analyze_288.py / verdict notebook can
    # recover hyperparameter trajectories without re-walking every gen dir.
    state_path = seed_root / "lineage_state.json"
    state_path.write_text(json.dumps([asdict(lin) for lin in lineages], indent=2))

    summary = {
        "pbt_seed": pbt_seed,
        "population_size": population_size,
        "generations": generations,
        "iters_per_gen": iters_per_gen,
        "final_best_trailing5": max(
            (lin.trailing5_team_history[-1] for lin in lineages),
            default=float("-inf"),
        ),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "experiments"
        / "p3_specialization"
        / "runs"
        / "issue288_pbt",
        help="Root output dir. Per-seed dirs land at <output-dir>/seed_<S>/.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="PBT-trial seeds. Each is an independent population.",
    )
    p.add_argument("--population-size", type=int, default=16)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--iters-per-gen", type=int, default=50)
    p.add_argument("--scenario", default="minimal_specialization")
    p.add_argument("--lambda-red", type=float, default=0.0)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--initial-lr", type=float, default=3e-4)
    p.add_argument("--initial-entropy-coef", type=float, default=0.01)
    p.add_argument(
        "--weight-noise",
        type=float,
        default=0.01,
        help=(
            "Std multiplier for additive Gaussian weight perturbation: noise = "
            "weight_noise * std(layer). Jaderberg-equivalent σ=0.01 keeps "
            "one mutation step from destroying a working policy."
        ),
    )
    p.add_argument(
        "--truncation-frac",
        type=float,
        default=0.25,
        help="Fraction of population replaced per generation (default 25%).",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for s in args.seeds:
        summary = run_pbt(
            output_root=args.output_dir,
            pbt_seed=s,
            population_size=args.population_size,
            generations=args.generations,
            iters_per_gen=args.iters_per_gen,
            scenario=args.scenario,
            lambda_red=args.lambda_red,
            num_agents=args.num_agents,
            rollout_steps=args.rollout_steps,
            initial_lr=args.initial_lr,
            initial_entropy_coef=args.initial_entropy_coef,
            weight_noise=args.weight_noise,
            truncation_frac=args.truncation_frac,
        )
        summaries.append(summary)
        print(f"\n== PBT seed {s} done. Summary: {summary}\n", flush=True)

    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"All PBT trials complete. Summary -> {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
