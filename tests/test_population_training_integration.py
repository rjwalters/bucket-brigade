"""
Integration tests for population-based training.

Tests the full training pipeline end-to-end with real multiprocessing,
GPU learners, and CPU simulator working together.

These tests are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import time

from bucket_brigade.training import PopulationTrainer
from bucket_brigade_core import SCENARIOS


@pytest.mark.slow
class TestPopulationTrainingBasic:
    """Basic integration tests for population training."""

    def test_small_population_training_run(self):
        """Test a small training run with minimal configuration.

        This validates the full pipeline:
        - Multiprocessing infrastructure setup
        - GPU learner processes spawn correctly
        - CPU simulator runs episodes
        - Experiences flow to learners
        - Policy updates flow to simulator
        """
        # Use temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Small configuration for fast test
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,  # Small population
                num_games=4,        # Few parallel games
                num_agents_per_game=4,
                hidden_size=64,     # Small network
                learning_rate=3e-4,
                device='cpu',       # Use CPU for test (works on CI)
                matchmaking_strategy='round_robin',
                seed=42,
                batch_size=32,      # Small batch
                num_epochs=2,       # Few epochs
                update_interval=10,
                checkpoint_dir=checkpoint_dir,
                log_interval=5,
            )

            # Train for just 20 episodes (enough to test pipeline)
            trainer.train(num_episodes=20)

            # Verify training completed
            assert trainer.total_episodes == 20

            # Verify simulator exists and has stats
            assert trainer.simulator is not None
            stats = trainer.simulator.get_statistics()
            assert stats['total_episodes'] == 20
            assert stats['total_steps'] > 0

            # Verify all agents got to play (round-robin)
            match_counts = stats['match_counts']
            assert len(match_counts) == 4
            # Each agent should have played multiple times
            assert all(count > 0 for count in match_counts)
            # Counts should be relatively balanced (within 2 of each other)
            assert max(match_counts) - min(match_counts) <= 2

    def test_population_training_with_checkpointing(self):
        """Test that checkpoint saving works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=checkpoint_dir,
            )

            # Train briefly
            trainer.train(num_episodes=10)

            # Save checkpoint
            checkpoint_path = checkpoint_dir / 'test_checkpoint.pt'
            trainer.save_checkpoint(checkpoint_path)

            # Verify checkpoint exists and contains expected keys
            assert checkpoint_path.exists()
            checkpoint = torch.load(checkpoint_path)

            assert 'scenario_name' in checkpoint
            assert 'population_size' in checkpoint
            assert 'total_episodes' in checkpoint
            assert 'policies' in checkpoint
            assert 'statistics' in checkpoint

            # Verify policies saved
            assert len(checkpoint['policies']) == 4
            for agent_id in range(4):
                assert agent_id in checkpoint['policies']
                # Each policy should be a state dict
                assert isinstance(checkpoint['policies'][agent_id], dict)

    def test_population_training_config_save(self):
        """Test that configuration saving works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                learning_rate=1e-3,
                device='cpu',
                seed=123,
                batch_size=16,
                checkpoint_dir=checkpoint_dir,
            )

            # Save config
            config_path = checkpoint_dir / 'config.json'
            trainer.save_config(config_path)

            # Verify config file exists
            assert config_path.exists()

            # Load and verify contents
            import json
            with open(config_path) as f:
                config = json.load(f)

            assert config['scenario_name'] == 'trivial_cooperation'
            assert config['population_size'] == 4
            assert config['num_games'] == 2
            assert config['hidden_size'] == 32
            assert config['learning_rate'] == pytest.approx(1e-3)
            assert config['seed'] == 123


@pytest.mark.slow
class TestPopulationTrainingMultiScenario:
    """Test population training across multiple scenarios."""

    @pytest.mark.parametrize('scenario_name', [
        'trivial_cooperation',
        'easy',
        'greedy_neighbor',
    ])
    def test_training_different_scenarios(self, scenario_name):
        """Test that training works with different scenarios.

        This is important for multi-scenario training runs.
        """
        # Skip if scenario doesn't exist
        if scenario_name not in SCENARIOS:
            pytest.skip(f"Scenario {scenario_name} not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            trainer = PopulationTrainer(
                scenario_name=scenario_name,
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=checkpoint_dir,
            )

            # Train for a few episodes
            trainer.train(num_episodes=5)

            # Verify completion
            assert trainer.total_episodes == 5
            stats = trainer.simulator.get_statistics()
            assert stats['total_episodes'] == 5
            assert stats['total_steps'] > 0


@pytest.mark.slow
class TestPopulationTrainingScaling:
    """Test population training with different scale configurations."""

    def test_larger_population(self):
        """Test with larger population (8 agents)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=8,  # Larger population
                num_games=4,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(num_episodes=10)

            # Verify all 8 agents trained
            stats = trainer.simulator.get_statistics()
            assert len(stats['match_counts']) == 8
            assert all(count > 0 for count in stats['match_counts'])

    def test_more_parallel_games(self):
        """Test with more parallel game environments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=8,  # More parallel games
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(num_episodes=10)

            # Should complete successfully
            assert trainer.total_episodes == 10


@pytest.mark.slow
class TestPopulationTrainingRobustness:
    """Test robustness and error handling."""

    def test_mismatched_population_size_validation(self):
        """Test that invalid population size is caught."""
        with pytest.raises(ValueError, match="Population size.*must be >= num_agents_per_game"):
            PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=2,       # Too small!
                num_agents_per_game=4,   # Needs 4 agents per game
                device='cpu',
            )

    def test_training_completes_without_hanging(self):
        """Test that training completes in reasonable time (no deadlocks).

        This is critical for long training runs - we don't want deadlocks!
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            # Set a timeout - if training takes >60 seconds for 5 episodes, something is wrong
            start_time = time.time()
            trainer.train(num_episodes=5)
            elapsed = time.time() - start_time

            # Should complete quickly (definitely under 60 seconds)
            assert elapsed < 60, f"Training took {elapsed:.1f}s - possible deadlock or performance issue"

    def test_cleanup_processes_on_completion(self):
        """Test that all processes are cleaned up after training."""
        import multiprocessing as mp

        # Get initial process count
        initial_processes = len(mp.active_children())

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(num_episodes=5)

            # Cleanup happens in cleanup() which is called by train()
            # Give processes a moment to fully terminate
            time.sleep(1)

        # All spawned processes should be cleaned up
        final_processes = len(mp.active_children())
        assert final_processes == initial_processes, \
            f"Process leak detected: started with {initial_processes}, ended with {final_processes}"


@pytest.mark.slow
class TestPopulationTrainingStatistics:
    """Test that statistics are correctly tracked."""

    def test_episode_rewards_tracked(self):
        """Test that episode rewards are tracked for all agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(num_episodes=10)

            stats = trainer.simulator.get_statistics()
            episode_rewards = stats['episode_rewards']

            # Should have rewards for all 4 agents
            assert len(episode_rewards) == 4

            # Each agent should have played at least once
            for agent_id, rewards in episode_rewards.items():
                assert len(rewards) > 0, f"Agent {agent_id} has no rewards recorded"

    def test_step_counting(self):
        """Test that total steps are counted correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(num_episodes=10)

            stats = trainer.simulator.get_statistics()

            # Should have taken some steps
            assert stats['total_steps'] > 0

            # Should have taken at least 1 step per episode
            assert stats['total_steps'] >= stats['total_episodes']

    def test_match_count_tracking(self):
        """Test that match counts are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=2,
                num_agents_per_game=4,
                hidden_size=32,
                device='cpu',
                matchmaking_strategy='round_robin',
                seed=42,
                batch_size=16,
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(num_episodes=12)  # 12 episodes, 4 agents per game = 3 games each

            stats = trainer.simulator.get_statistics()
            match_counts = stats['match_counts']

            # All agents should have played 3 times (round-robin, 12 episodes ÷ 4 agents)
            assert all(count == 3 for count in match_counts), \
                f"Round-robin failed: {match_counts}"


@pytest.mark.slow
class TestPopulationTrainingEndToEnd:
    """Full end-to-end integration test simulating a real training run."""

    def test_full_training_pipeline_simulation(self):
        """
        Comprehensive test simulating a realistic (but small) training run.

        This test validates:
        - Full pipeline setup
        - Multi-episode training
        - Checkpoint saving mid-training
        - Statistics collection
        - Proper cleanup
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Configuration similar to actual training (but scaled down)
            trainer = PopulationTrainer(
                scenario_name='trivial_cooperation',
                population_size=4,
                num_games=4,
                num_agents_per_game=4,
                hidden_size=64,
                learning_rate=3e-4,
                device='cpu',
                matchmaking_strategy='round_robin',
                seed=42,
                batch_size=32,
                num_epochs=2,
                update_interval=5,
                checkpoint_dir=checkpoint_dir,
                log_interval=5,
            )

            # Train for multiple batches of episodes
            trainer.train(num_episodes=25)

            # Verify training metrics
            assert trainer.total_episodes == 25
            stats = trainer.simulator.get_statistics()
            assert stats['total_episodes'] == 25
            assert stats['total_steps'] > 25  # Should have multiple steps per episode

            # Save final checkpoint
            final_checkpoint = checkpoint_dir / 'final.pt'
            trainer.save_checkpoint(final_checkpoint)
            assert final_checkpoint.exists()

            # Verify checkpoint contents
            checkpoint = torch.load(final_checkpoint)
            assert checkpoint['total_episodes'] == 25
            assert len(checkpoint['policies']) == 4

            # Verify all policies have reasonable state
            for agent_id, state_dict in checkpoint['policies'].items():
                assert len(state_dict) > 0
                # Check that weights are finite (not NaN or Inf)
                for param_name, param in state_dict.items():
                    assert torch.isfinite(param).all(), \
                        f"Agent {agent_id} has non-finite weights in {param_name}"

            # Save config
            config_path = checkpoint_dir / 'config.json'
            trainer.save_config(config_path)
            assert config_path.exists()

            # Verify match distribution is balanced
            match_counts = stats['match_counts']
            max_diff = max(match_counts) - min(match_counts)
            # With round-robin and 25 episodes, difference should be at most 1
            assert max_diff <= 1, f"Unbalanced matchmaking: {match_counts}"
