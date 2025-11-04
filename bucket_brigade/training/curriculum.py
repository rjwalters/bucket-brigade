"""Curriculum learning orchestration for progressive training.

This module provides infrastructure for curriculum-based reinforcement learning,
where agents progressively learn from simpler to more complex scenarios.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from bucket_brigade.envs import get_scenario_by_name
from bucket_brigade.envs.puffer_env import PufferBucketBrigade

from .networks import PolicyNetwork


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage.

    Attributes:
        name: Scenario name for this stage
        num_opponents: Number of opponent agents
        steps: Training steps for this stage
        description: Human-readable description of what this stage teaches
        progression_threshold: Mean reward threshold to advance to next stage
    """

    name: str
    num_opponents: int
    steps: int
    description: str
    progression_threshold: float


class CurriculumTrainer:
    """Manages progressive training through a curriculum of scenarios.

    The curriculum starts with simple cooperation scenarios and progressively
    introduces more complex strategic challenges. Each stage builds on skills
    learned in previous stages.

    Args:
        args: Training configuration arguments containing hyperparameters,
            curriculum settings, and run configuration

    Example:
        >>> from argparse import Namespace
        >>> args = Namespace(
        ...     run_name="my_curriculum",
        ...     hidden_size=64,
        ...     lr=3e-4,
        ...     batch_size=2048,
        ...     num_epochs=4,
        ...     custom_curriculum=False,
        ...     # ... other hyperparameters
        ... )
        >>> trainer = CurriculumTrainer(args)
        >>> trainer.train_curriculum()
    """

    # Default curriculum stages (easier to harder)
    DEFAULT_CURRICULUM = [
        {
            "name": "trivial_cooperation",
            "num_opponents": 2,
            "steps": 100_000,
            "description": "Learn basic cooperation",
            "progression_threshold": 5.0,
        },
        {
            "name": "early_containment",
            "num_opponents": 3,
            "steps": 150_000,
            "description": "Learn timing and coordination",
            "progression_threshold": 4.0,
        },
        {
            "name": "greedy_neighbor",
            "num_opponents": 3,
            "steps": 200_000,
            "description": "Learn social dilemmas",
            "progression_threshold": 3.0,
        },
        {
            "name": "sparse_heroics",
            "num_opponents": 4,
            "steps": 250_000,
            "description": "Learn resource allocation",
            "progression_threshold": 2.5,
        },
        {
            "name": "chain_reaction",
            "num_opponents": 4,
            "steps": 300_000,
            "description": "Learn distributed coordination",
            "progression_threshold": 2.0,
        },
    ]

    def __init__(self, args: Any) -> None:
        """Initialize curriculum trainer with configuration.

        Args:
            args: Namespace or object with training configuration attributes
        """
        self.args = args

        # Use default curriculum or allow custom override
        self.curriculum = self.DEFAULT_CURRICULUM.copy()

        if hasattr(args, "custom_curriculum") and args.custom_curriculum:
            print("Using custom curriculum from arguments")
            # User can override stages via arguments if needed
            # This is a hook for future extensibility

    def train_curriculum(self) -> None:
        """Train through the full curriculum of scenarios.

        This method:
        1. Initializes the policy network and optimizer
        2. Trains through each curriculum stage sequentially
        3. Evaluates performance at each stage
        4. Saves checkpoints after each stage
        5. Performs final evaluation across all scenarios
        """
        print("<� Starting Curriculum Training")
        print("=" * 60)

        # Initialize TensorBoard logging
        run_name = (
            self.args.run_name
            if hasattr(self.args, "run_name") and self.args.run_name
            else f"curriculum_{int(time.time())}"
        )
        writer = SummaryWriter(f"runs/{run_name}")
        print(f"=� TensorBoard logging to: runs/{run_name}")

        # Create initial environment to get observation/action space dimensions
        initial_scenario = get_scenario_by_name(
            self.curriculum[0]["name"],
            num_agents=self.curriculum[0]["num_opponents"] + 1,
        )
        initial_env = PufferBucketBrigade(
            scenario=initial_scenario,
            num_opponents=self.curriculum[0]["num_opponents"],
        )

        obs_dim = initial_env.observation_space.shape[0]
        action_dims = initial_env.action_space.nvec.tolist()

        # Initialize policy network
        print(">� Creating policy network")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dims: {action_dims}")

        hidden_size = getattr(self.args, "hidden_size", 64)
        policy = PolicyNetwork(obs_dim, action_dims, hidden_size=hidden_size)
        optimizer = optim.Adam(policy.parameters(), lr=self.args.lr)

        global_step = 0
        start_time = time.time()

        # Train through curriculum stages
        for stage_idx, stage_dict in enumerate(self.curriculum):
            stage = (
                CurriculumStage(**stage_dict)
                if isinstance(stage_dict, dict)
                else stage_dict
            )

            print(f"\n{'=' * 60}")
            print(f"=� Stage {stage_idx + 1}/{len(self.curriculum)}: {stage.name}")
            print(f"   {stage.description}")
            print(f"   Steps: {stage.steps:,}")
            print(f"   Opponents: {stage.num_opponents}")
            print(f"   Progression threshold: {stage.progression_threshold:.2f}")
            print(f"{'=' * 60}")

            # Create environment for this stage
            scenario = get_scenario_by_name(
                stage.name, num_agents=stage.num_opponents + 1
            )
            env = PufferBucketBrigade(
                scenario=scenario,
                num_opponents=stage.num_opponents,
            )

            # Import training function (still in parent script for now)
            # This maintains Phase 1 "lighter refactoring" scope
            from scripts.train_curriculum import train_stage_ppo, evaluate_policy

            # Train on this stage
            stage_start = time.time()
            global_step, final_reward = train_stage_ppo(
                env=env,
                policy=policy,
                optimizer=optimizer,
                num_steps=stage.steps,
                batch_size=self.args.batch_size,
                num_epochs=self.args.num_epochs,
                clip_epsilon=self.args.clip_epsilon,
                value_coef=self.args.value_coef,
                entropy_coef=self.args.entropy_coef,
                max_grad_norm=self.args.max_grad_norm,
                writer=writer,
                global_step_offset=global_step,
            )
            stage_duration = time.time() - stage_start

            # Evaluate on this stage
            eval_reward = evaluate_policy(policy, env, num_episodes=20)
            print("\n    Stage complete!")
            print(f"   Training reward: {final_reward:.2f}")
            print(f"   Evaluation reward: {eval_reward:.2f}")
            print(f"   Duration: {stage_duration / 60:.1f} minutes")

            # Log stage completion
            writer.add_scalar(
                f"curriculum/stage_{stage_idx}_eval", eval_reward, global_step
            )
            writer.add_scalar(
                f"curriculum/stage_{stage_idx}_train", final_reward, global_step
            )

            # Save stage checkpoint
            self._save_checkpoint(
                policy=policy,
                obs_dim=obs_dim,
                action_dims=action_dims,
                hidden_size=hidden_size,
                stage_idx=stage_idx,
                stage_name=stage.name,
                run_name=run_name,
            )

            # Check progression threshold
            if (
                eval_reward < stage.progression_threshold
                and stage_idx < len(self.curriculum) - 1
            ):
                print(
                    f"   �  Warning: Performance below threshold "
                    f"({eval_reward:.2f} < {stage.progression_threshold:.2f})"
                )
                print(
                    "   Continuing to next stage anyway "
                    "(human approval recommended)"
                )

        # Final evaluation across all scenarios
        self._final_evaluation(
            policy=policy, writer=writer, global_step=global_step
        )

        # Save final model
        output_dir = Path("models") / run_name
        final_path = output_dir / "curriculum_final.pt"
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dims": action_dims,
                "hidden_size": hidden_size,
            },
            final_path,
        )

        total_duration = time.time() - start_time
        print(f"\n{'=' * 60}")
        print("<� Curriculum Training Complete!")
        print(f"   Total duration: {total_duration / 60:.1f} minutes")
        print(f"   Final model: {final_path}")
        print(f"   TensorBoard: runs/{run_name}")
        print(f"{'=' * 60}")

        writer.close()

    def _save_checkpoint(
        self,
        policy: PolicyNetwork,
        obs_dim: int,
        action_dims: List[int],
        hidden_size: int,
        stage_idx: int,
        stage_name: str,
        run_name: str,
    ) -> None:
        """Save checkpoint after completing a curriculum stage.

        Args:
            policy: Trained policy network
            obs_dim: Observation space dimension
            action_dims: Action space dimensions
            hidden_size: Hidden layer size
            stage_idx: Index of completed stage
            stage_name: Name of completed stage
            run_name: Name of this training run
        """
        output_dir = Path("models") / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        stage_path = output_dir / f"stage_{stage_idx}_{stage_name}.pt"
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dims": action_dims,
                "hidden_size": hidden_size,
                "stage": stage_idx,
                "stage_name": stage_name,
            },
            stage_path,
        )
        print(f"   =� Saved checkpoint: {stage_path}")

    def _final_evaluation(
        self,
        policy: PolicyNetwork,
        writer: SummaryWriter,
        global_step: int,
    ) -> None:
        """Evaluate final policy across all curriculum scenarios.

        Args:
            policy: Trained policy network
            writer: TensorBoard writer for logging
            global_step: Current global training step
        """
        # Import evaluation function (still in parent script for now)
        from scripts.train_curriculum import evaluate_policy

        print(f"\n{'=' * 60}")
        print("<� Final Evaluation Across All Scenarios")
        print(f"{'=' * 60}")

        for stage_dict in self.curriculum:
            stage = (
                CurriculumStage(**stage_dict)
                if isinstance(stage_dict, dict)
                else stage_dict
            )

            scenario = get_scenario_by_name(
                stage.name, num_agents=stage.num_opponents + 1
            )
            env = PufferBucketBrigade(
                scenario=scenario, num_opponents=stage.num_opponents
            )
            eval_reward = evaluate_policy(policy, env, num_episodes=30)

            print(f"   {stage.name:20s}: {eval_reward:.2f}")
            writer.add_scalar(f"final_eval/{stage.name}", eval_reward, global_step)
