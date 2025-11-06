"""
Population-Based Training Coordinator.

This module coordinates population-based training by:
1. Setting up multiprocessing infrastructure (queues, processes)
2. Spawning GPU learner processes
3. Running CPU game simulator
4. Monitoring overall training progress
5. Saving checkpoints and logging metrics

Architecture:
    PopulationTrainer (Coordinator)
        ↓ Spawns
    [GPU Learner 0, GPU Learner 1, ..., GPU Learner N] (Processes)
        ↓ Creates
    GameSimulator (CPU Process)
        ↓ Coordinates via Queues
    Experience flow: Simulator → Learners
    Policy flow: Learners → Simulator
"""

import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict
import time
import json

import torch
import numpy as np

from bucket_brigade_core import SCENARIOS, Scenario
from bucket_brigade.training import PolicyNetwork
from bucket_brigade.training.game_simulator import GameSimulator
from bucket_brigade.training.policy_learner import learner_process
from bucket_brigade.training.observation_utils import flatten_observation, get_observation_dim, create_scenario_info


class PopulationTrainer:
    """
    Coordinator for population-based training.

    Manages the full training pipeline with CPU simulator and multiple GPU learners.
    """

    def __init__(
        self,
        scenario_name: str,
        population_size: int = 8,
        num_games: int = 64,
        num_agents_per_game: int = 4,
        # Network architecture
        hidden_size: int = 512,
        learning_rate: float = 3e-4,
        # Training parameters
        device: str = "cuda",
        matchmaking_strategy: str = "round_robin",
        seed: Optional[int] = None,
        # Learner parameters
        batch_size: int = 256,
        num_epochs: int = 4,
        update_interval: int = 100,
        # Checkpoint and logging
        checkpoint_dir: Optional[Path] = None,
        log_interval: int = 100,
    ):
        """
        Initialize the population trainer.

        Args:
            scenario_name: Name of the scenario to train on
            population_size: Number of agents in the population
            num_games: Number of parallel game environments
            num_agents_per_game: Number of agents per game (typically 4)
            hidden_size: Hidden layer size for policy networks
            learning_rate: Learning rate for optimization
            device: Device for GPU learners ("cuda" or "cpu")
            matchmaking_strategy: Strategy for agent matchmaking
            seed: Random seed
            batch_size: Batch size for learners
            num_epochs: PPO epochs per batch
            update_interval: Learners send policy updates every N batches
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Log progress every N simulation episodes
        """
        self.scenario_name = scenario_name
        self.population_size = population_size
        self.num_games = num_games
        self.num_agents_per_game = num_agents_per_game
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = device
        self.matchmaking_strategy = matchmaking_strategy
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.update_interval = update_interval
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval

        # Get scenario
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        self.scenario = SCENARIOS[scenario_name]

        # Validate population size
        if population_size < num_agents_per_game:
            raise ValueError(
                f"Population size ({population_size}) must be >= num_agents_per_game ({num_agents_per_game}). "
                f"Each game needs {num_agents_per_game} agents, so the population must have at least that many."
            )

        # Initialize dimensions (from scenario)
        # For now, we'll infer these from a test environment
        from bucket_brigade_core import BucketBrigade
        test_env = BucketBrigade(self.scenario, num_agents_per_game, seed=seed)

        # Get number of houses from a test observation
        test_obs_obj = test_env.get_observation(0)
        num_houses = len(test_obs_obj.houses)

        # Calculate observation dimension
        self.obs_dim = get_observation_dim(num_houses, num_agents_per_game)

        # Action dimensions: [num_houses, 3 modes (idle, fill, pass)]
        self.action_dims = [num_houses, 3]

        print(f"Inferred dimensions:")
        print(f"  Observation: {self.obs_dim}")
        print(f"  Actions: {self.action_dims} (houses={num_houses}, modes=3)")

        # Multiprocessing queues
        self.experience_queues: List[mp.Queue] = []
        self.policy_update_queue: Optional[mp.Queue] = None

        # GPU learner processes
        self.learner_processes: List[mp.Process] = []

        # Game simulator
        self.simulator: Optional[GameSimulator] = None

        # Statistics
        self.start_time = None
        self.total_episodes = 0

    def setup_infrastructure(self):
        """Set up multiprocessing queues and processes."""
        print(f"\n{'='*60}")
        print(f"🔧 Setting Up Training Infrastructure")
        print(f"{'='*60}")

        # Create experience queues (one per agent)
        print(f"Creating {self.population_size} experience queues...")
        self.experience_queues = [mp.Queue(maxsize=1000) for _ in range(self.population_size)]

        # Create policy update queue (shared)
        print("Creating policy update queue...")
        self.policy_update_queue = mp.Queue(maxsize=100)

        print(f"✅ Queues created\n")

    def spawn_learners(self):
        """Spawn GPU learner processes."""
        print(f"🚀 Spawning {self.population_size} GPU Learner Processes")
        print(f"{'='*60}")

        for agent_id in range(self.population_size):
            # Determine device for this learner
            # For multi-GPU, could use: cuda:{agent_id % num_gpus}
            learner_device = self.device

            print(f"Spawning learner for Agent {agent_id} on {learner_device}...")

            process = mp.Process(
                target=learner_process,
                kwargs={
                    'agent_id': agent_id,
                    'obs_dim': self.obs_dim,
                    'action_dims': self.action_dims,
                    'hidden_size': self.hidden_size,
                    'learning_rate': self.learning_rate,
                    'device': learner_device,
                    'experience_queue': self.experience_queues[agent_id],
                    'policy_update_queue': self.policy_update_queue,
                    'batch_size': self.batch_size,
                    'num_epochs': self.num_epochs,
                    'update_interval': self.update_interval,
                }
            )
            process.start()
            self.learner_processes.append(process)

        print(f"✅ All {self.population_size} learners spawned\n")

    def initialize_simulator(self):
        """Initialize the CPU game simulator."""
        print(f"🎮 Initializing CPU Game Simulator")
        print(f"{'='*60}")

        # Create simulator
        self.simulator = GameSimulator(
            scenario=self.scenario,
            num_games=self.num_games,
            population_size=self.population_size,
            num_agents_per_game=self.num_agents_per_game,
            experience_queues=self.experience_queues,
            policy_update_queue=self.policy_update_queue,
            matchmaking_strategy=self.matchmaking_strategy,
            seed=self.seed,
        )

        # Register initial policies for all agents
        print(f"Creating initial policies for {self.population_size} agents...")
        for agent_id in range(self.population_size):
            policy = PolicyNetwork(
                obs_dim=self.obs_dim,
                action_dims=self.action_dims,
                hidden_size=self.hidden_size,
            )
            # Randomize initial weights slightly for diversity
            if self.seed is not None:
                torch.manual_seed(self.seed + agent_id)
            self.simulator.register_policy(agent_id, policy)

        print(f"✅ Simulator initialized\n")

    def train(self, num_episodes: int):
        """
        Run the full population training.

        Args:
            num_episodes: Number of episodes to simulate
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Population-Based Training")
        print(f"{'='*60}")
        print(f"Scenario: {self.scenario_name}")
        print(f"Population: {self.population_size} agents")
        print(f"Parallel games: {self.num_games}")
        print(f"Target episodes: {num_episodes:,}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        self.start_time = time.time()

        # Setup infrastructure
        self.setup_infrastructure()

        # Spawn GPU learners
        self.spawn_learners()

        # Give learners time to initialize
        print("Waiting for learners to initialize...")
        time.sleep(3)

        # Initialize simulator
        self.initialize_simulator()

        # Run simulation
        print(f"▶️  Starting game simulation...")
        self.simulator.run(
            num_episodes=num_episodes,
            update_interval=self.log_interval,
        )

        # Training complete
        self.total_episodes = num_episodes
        self.cleanup()

    def cleanup(self):
        """Clean up processes and resources."""
        print(f"\n{'='*60}")
        print(f"🧹 Cleaning Up")
        print(f"{'='*60}")

        # Terminate learner processes
        print(f"Terminating {len(self.learner_processes)} learner processes...")
        for i, process in enumerate(self.learner_processes):
            print(f"  Terminating learner {i}...")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                print(f"  Force killing learner {i}...")
                process.kill()

        print("✅ All processes terminated\n")

        # Print final statistics
        if self.simulator:
            stats = self.simulator.get_statistics()
            total_time = time.time() - self.start_time

            print(f"{'='*60}")
            print(f"📊 Final Statistics")
            print(f"{'='*60}")
            print(f"Total episodes: {stats['total_episodes']:,}")
            print(f"Total steps: {stats['total_steps']:,}")
            print(f"Training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
            print(f"Episodes/sec: {stats['total_episodes']/total_time:.1f}")
            print(f"\nMatch counts per agent:")
            for agent_id, count in enumerate(stats['match_counts']):
                print(f"  Agent {agent_id}: {count:,} games")
            print(f"{'='*60}\n")

    def save_checkpoint(self, path: Path):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        if not self.simulator:
            print("Cannot save checkpoint - simulator not initialized")
            return

        checkpoint = {
            'scenario_name': self.scenario_name,
            'population_size': self.population_size,
            'total_episodes': self.total_episodes,
            'statistics': self.simulator.get_statistics(),
            'policies': {
                agent_id: policy.state_dict()
                for agent_id, policy in self.simulator.policies.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")

    def save_config(self, path: Path):
        """Save training configuration."""
        config = {
            'scenario_name': self.scenario_name,
            'population_size': self.population_size,
            'num_games': self.num_games,
            'num_agents_per_game': self.num_agents_per_game,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'matchmaking_strategy': self.matchmaking_strategy,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'update_interval': self.update_interval,
            'obs_dim': self.obs_dim,
            'action_dims': self.action_dims,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Saved config: {path}")
