"""
Unit tests for policy learner (GPU-side PPO training).

Tests the GPU-based policy learner that trains agents using PPO.
Focuses heavily on GAE computation where we found bugs during GPU testing.
"""

import pytest
import numpy as np
import torch
import multiprocessing as mp
from unittest.mock import Mock, MagicMock, patch
from collections import deque

from bucket_brigade.training.policy_learner import PolicyLearner
from bucket_brigade.training import compute_gae


class TestPolicyLearnerInitialization:
    """Tests for PolicyLearner initialization."""

    def test_basic_initialization(self):
        """Test basic learner initialization."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            learning_rate=3e-4,
            device="cpu",
        )

        assert learner.agent_id == 0
        assert learner.device == torch.device("cpu")
        assert learner.batch_size == 256
        assert learner.num_epochs == 4
        assert learner.total_batches == 0
        assert learner.total_updates == 0

    def test_initialization_with_queues(self):
        """Test initialization with multiprocessing queues."""
        exp_queue = Mock()
        update_queue = Mock()

        learner = PolicyLearner(
            agent_id=1,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            experience_queue=exp_queue,
            policy_update_queue=update_queue,
        )

        assert learner.experience_queue is exp_queue
        assert learner.policy_update_queue is update_queue

    def test_ppo_hyperparameters(self):
        """Test PPO hyperparameter configuration."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            gamma=0.98,
            gae_lambda=0.97,
            clip_epsilon=0.3,
            value_coef=0.6,
            entropy_coef=0.02,
        )

        assert learner.gamma == pytest.approx(0.98)
        assert learner.gae_lambda == pytest.approx(0.97)
        assert learner.clip_epsilon == pytest.approx(0.3)
        assert learner.value_coef == pytest.approx(0.6)
        assert learner.entropy_coef == pytest.approx(0.02)

    def test_policy_network_created(self):
        """Test that policy network is created correctly."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=256,
            device="cpu",
        )

        assert learner.policy is not None
        assert isinstance(learner.policy, torch.nn.Module)

        # Test forward pass
        obs = torch.randn(1, 36)
        action_logits, value = learner.policy(obs)
        assert len(action_logits) == 2
        assert action_logits[0].shape == (1, 10)
        assert action_logits[1].shape == (1, 3)
        assert value.shape == (1, 1)

    def test_experience_buffer_initialized(self):
        """Test that experience buffer is initialized."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=128,
        )

        assert isinstance(learner.experience_buffer, deque)
        assert learner.experience_buffer.maxlen == 256  # batch_size * 2


class TestGAEComputation:
    """Tests for GAE (Generalized Advantage Estimation) computation.

    This is CRITICAL - we found bugs here during GPU testing!
    """

    def test_compute_gae_basic(self):
        """Test basic GAE computation."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [0.5, 1.0, 1.5, 2.0, 2.5]
        dones = [False, False, False, False, True]

        advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Verify output type and length
        assert isinstance(advantages, list)
        assert len(advantages) == 5

        # Advantages should be floats
        assert all(isinstance(adv, float) for adv in advantages)

    def test_compute_gae_all_done(self):
        """Test GAE with all episodes done."""
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        dones = [True, True, True]

        advantages = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        # Each advantage should just be the reward (no bootstrapping)
        assert all(adv == pytest.approx(1.0) for adv in advantages)

    def test_compute_gae_no_done(self):
        """Test GAE with no terminal states."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5, 0.5]
        dones = [False, False, False, False]

        advantages = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        assert len(advantages) == 4
        # Later advantages should incorporate future rewards
        assert advantages[-1] < advantages[0]  # Discounting effect

    def test_compute_gae_returns_list_not_tensor(self):
        """REGRESSION TEST: compute_gae returns list, not tensor.

        This was a bug we found - trying to call .to(device) on the result.
        """
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, True]

        result = compute_gae(rewards, values, dones)

        # Must be a list (not tensor!)
        assert isinstance(result, list)
        assert not isinstance(result, torch.Tensor)

    def test_compute_gae_correct_number_of_args(self):
        """REGRESSION TEST: compute_gae takes 3-5 args, not 6.

        This was a bug - we were passing next_values which doesn't exist.
        """
        rewards = [1.0, 2.0]
        values = [0.5, 1.0]
        dones = [False, True]

        # Should work with 3 args (+ optional gamma, gae_lambda)
        result = compute_gae(rewards, values, dones)
        assert len(result) == 2

        # Should work with all 5 args
        result = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        assert len(result) == 2

        # Verify signature doesn't accept next_values
        with pytest.raises(TypeError):
            compute_gae(rewards, values, dones, [1.0, 2.0], 0.99, 0.95)


class TestBatchPreparation:
    """Tests for batch preparation from experience buffer."""

    def test_prepare_batch_basic(self):
        """Test basic batch preparation."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=4,
        )

        # Add experiences to buffer
        for i in range(4):
            exp = {
                'obs': np.random.randn(36).astype(np.float32),
                'action': [i % 10, i % 3],
                'reward': float(i),
                'next_obs': np.random.randn(36).astype(np.float32),
                'done': False,
                'logprob': -0.5,
            }
            learner.experience_buffer.append(exp)

        # Prepare batch
        obs, actions, rewards, next_obs, dones, old_logprobs = learner.prepare_batch()

        # Verify shapes
        assert obs.shape == (4, 36)
        assert actions.shape == (4, 2)
        assert rewards.shape == (4,)
        assert next_obs.shape == (4, 36)
        assert dones.shape == (4,)
        assert old_logprobs.shape == (4,)

        # Verify types
        assert obs.dtype == torch.float32
        assert actions.dtype == torch.long
        assert rewards.dtype == torch.float32

    def test_prepare_batch_with_done_states(self):
        """Test batch preparation with terminal states."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=3,
        )

        # Add experiences with some done=True
        for i in range(3):
            exp = {
                'obs': np.random.randn(36).astype(np.float32),
                'action': [0, 0],
                'reward': 1.0,
                'next_obs': np.random.randn(36).astype(np.float32) if i < 2 else None,
                'done': i == 2,
                'logprob': -0.5,
            }
            learner.experience_buffer.append(exp)

        obs, actions, rewards, next_obs, dones, old_logprobs = learner.prepare_batch()

        # Last state should be marked as done
        assert dones[2] == 1.0
        assert dones[0] == 0.0
        assert dones[1] == 0.0

        # Next obs for done state should be zeros
        assert torch.allclose(next_obs[2], torch.zeros(36))

    def test_prepare_batch_smaller_than_batch_size(self):
        """Test batch preparation when buffer has fewer items."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=10,
        )

        # Add only 3 experiences
        for i in range(3):
            exp = {
                'obs': np.zeros(36, dtype=np.float32),
                'action': [0, 0],
                'reward': 1.0,
                'next_obs': np.zeros(36, dtype=np.float32),
                'done': False,
                'logprob': -0.5,
            }
            learner.experience_buffer.append(exp)

        obs, actions, rewards, next_obs, dones, old_logprobs = learner.prepare_batch()

        # Should only prepare 3 items
        assert obs.shape[0] == 3


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_compute_ppo_loss_shapes(self):
        """Test that PPO loss computation returns correct shapes."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
        )

        # Create dummy batch
        batch_size = 8
        observations = torch.randn(batch_size, 36)
        # Actions: [house (0-9), mode (0-2)]
        house_actions = torch.randint(0, 10, (batch_size,))
        mode_actions = torch.randint(0, 3, (batch_size,))
        actions = torch.stack([house_actions, mode_actions], dim=1)
        old_logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        # Compute loss
        total_loss, policy_loss, value_loss, entropy = learner.compute_ppo_loss(
            observations, actions, old_logprobs, advantages, returns
        )

        # Verify all are scalars
        assert total_loss.shape == ()
        assert policy_loss.shape == ()
        assert value_loss.shape == ()
        assert entropy.shape == ()

    def test_compute_ppo_loss_values_reasonable(self):
        """Test that PPO loss values are in reasonable ranges."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            clip_epsilon=0.2,
        )

        batch_size = 16
        observations = torch.randn(batch_size, 36)
        # Actions: [house (0-9), mode (0-2)]
        house_actions = torch.randint(0, 10, (batch_size,))
        mode_actions = torch.randint(0, 3, (batch_size,))
        actions = torch.stack([house_actions, mode_actions], dim=1)
        old_logprobs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)

        total_loss, policy_loss, value_loss, entropy = learner.compute_ppo_loss(
            observations, actions, old_logprobs, advantages, returns
        )

        # Losses should be finite
        assert torch.isfinite(total_loss)
        assert torch.isfinite(policy_loss)
        assert torch.isfinite(value_loss)
        assert torch.isfinite(entropy)

        # Entropy should be positive (indicates exploration)
        assert entropy.item() > 0


class TestTrainBatch:
    """Tests for training on a batch."""

    def test_train_batch_basic(self):
        """Test basic batch training."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=8,
            num_epochs=2,
        )

        # Add experiences
        for i in range(8):
            exp = {
                'obs': np.random.randn(36).astype(np.float32),
                'action': [i % 10, i % 3],
                'reward': np.random.rand(),
                'next_obs': np.random.randn(36).astype(np.float32),
                'done': False,
                'logprob': np.random.randn(),
            }
            learner.experience_buffer.append(exp)

        # Train
        stats = learner.train_batch()

        # Verify statistics returned
        assert 'batch' in stats
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy' in stats
        assert 'mean_reward' in stats

        # Verify batch counter incremented
        assert learner.total_batches == 1

    def test_train_batch_updates_weights(self):
        """Test that training actually updates network weights."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=4,
            num_epochs=1,
        )

        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in learner.policy.named_parameters()
        }

        # Add experiences
        for i in range(4):
            exp = {
                'obs': np.random.randn(36).astype(np.float32),
                'action': [0, 0],
                'reward': 1.0,
                'next_obs': np.random.randn(36).astype(np.float32),
                'done': False,
                'logprob': -2.0,
            }
            learner.experience_buffer.append(exp)

        # Train
        learner.train_batch()

        # Verify weights changed
        for name, param in learner.policy.named_parameters():
            # At least some parameters should have changed
            if 'weight' in name or 'bias' in name:
                # We expect at least one param to change (use any() to be lenient)
                pass

        # Check that at least one parameter changed significantly
        changed = False
        for name, param in learner.policy.named_parameters():
            if not torch.allclose(param, initial_weights[name], atol=1e-6):
                changed = True
                break
        assert changed, "No parameters were updated during training"


class TestExperienceCollection:
    """Tests for collecting experiences from queue."""

    def test_collect_experiences_basic(self):
        """Test basic experience collection."""
        # Create mock queue
        exp_queue = Mock()

        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            experience_queue=exp_queue,
            batch_size=5,
        )

        # Configure queue to return experiences for this agent
        experiences = []
        for i in range(5):
            exp = {
                'obs': np.zeros(36),
                'action': [0, 0],
                'reward': 1.0,
                'next_obs': np.zeros(36),
                'done': False,
                'logprob': -1.0,
            }
            experiences.append((0, exp))

        # Mock queue behavior
        exp_queue.empty.side_effect = [False] * 5 + [True]
        exp_queue.get.side_effect = experiences

        # Collect
        success = learner.collect_experiences(min_size=5)

        assert success
        assert len(learner.experience_buffer) == 5

    def test_collect_experiences_filters_by_agent_id(self):
        """Test that experiences are filtered by agent ID (on-policy)."""
        exp_queue = Mock()

        learner = PolicyLearner(
            agent_id=2,  # Agent 2
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            experience_queue=exp_queue,
            batch_size=5,
        )

        # Mix of experiences for different agents
        experiences = [
            (0, {'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0, 'next_obs': np.zeros(36), 'done': False, 'logprob': -1.0}),
            (2, {'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0, 'next_obs': np.zeros(36), 'done': False, 'logprob': -1.0}),
            (1, {'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0, 'next_obs': np.zeros(36), 'done': False, 'logprob': -1.0}),
            (2, {'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0, 'next_obs': np.zeros(36), 'done': False, 'logprob': -1.0}),
            (2, {'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0, 'next_obs': np.zeros(36), 'done': False, 'logprob': -1.0}),
        ]

        exp_queue.empty.side_effect = [False] * 5 + [True]
        exp_queue.get.side_effect = experiences

        # Collect 3 experiences for agent 2
        success = learner.collect_experiences(min_size=3)

        assert success
        # Should only have agent 2's experiences
        assert len(learner.experience_buffer) == 3


class TestPolicyUpdateSending:
    """Tests for sending policy updates to CPU simulator."""

    def test_send_policy_update_basic(self):
        """Test sending policy update."""
        update_queue = Mock()

        learner = PolicyLearner(
            agent_id=3,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            policy_update_queue=update_queue,
        )

        # Send update
        learner.send_policy_update()

        # Verify queue.put was called
        assert update_queue.put.called

        # Verify content
        call_args = update_queue.put.call_args[0][0]
        agent_id, state_dict = call_args
        assert agent_id == 3
        assert isinstance(state_dict, dict)

        # Verify update counter incremented
        assert learner.total_updates == 1

    def test_send_policy_update_no_queue(self):
        """Test that sending update with no queue doesn't crash."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            policy_update_queue=None,
        )

        # Should not raise error
        learner.send_policy_update()


class TestStatistics:
    """Tests for statistics tracking."""

    def test_get_statistics_initial(self):
        """Test getting statistics from fresh learner."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
        )

        stats = learner.get_statistics()

        assert stats['total_batches'] == 0
        assert stats['total_updates'] == 0
        assert len(stats['policy_loss_history']) == 0
        assert len(stats['value_loss_history']) == 0
        assert len(stats['entropy_history']) == 0

    def test_statistics_tracked_during_training(self):
        """Test that statistics are updated during training."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
            batch_size=4,
        )

        # Add experiences and train
        for i in range(4):
            exp = {
                'obs': np.random.randn(36).astype(np.float32),
                'action': [0, 0],
                'reward': 1.0,
                'next_obs': np.random.randn(36).astype(np.float32),
                'done': False,
                'logprob': -1.0,
            }
            learner.experience_buffer.append(exp)

        learner.train_batch()

        stats = learner.get_statistics()

        assert stats['total_batches'] == 1
        assert len(stats['policy_loss_history']) == 1
        assert len(stats['value_loss_history']) == 1
        assert len(stats['entropy_history']) == 1


class TestGetPolicyState:
    """Tests for getting policy state dict."""

    def test_get_policy_state_returns_dict(self):
        """Test that get_policy_state returns a state dict."""
        learner = PolicyLearner(
            agent_id=0,
            obs_dim=36,
            action_dims=[10, 3],
            hidden_size=128,
            device="cpu",
        )

        state = learner.get_policy_state()

        assert isinstance(state, dict)
        assert len(state) > 0

        # Should contain typical network parameters
        keys = list(state.keys())
        assert any('weight' in key for key in keys)
        assert any('bias' in key for key in keys)
