"""Tests for the training module.

This module tests the PolicyNetwork and CurriculumTrainer classes to ensure
they work correctly after refactoring from the monolithic train_curriculum.py script.
"""

import torch

from bucket_brigade.training import PolicyNetwork


class TestPolicyNetwork:
    """Test suite for PolicyNetwork class."""

    def test_forward_pass_shape(self):
        """Test that forward pass produces expected output shapes."""
        obs_dim = 42
        action_dims = [10, 2]
        hidden_size = 64
        batch_size = 16

        policy = PolicyNetwork(obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size)
        obs = torch.randn(batch_size, obs_dim)

        action_logits, value = policy(obs)

        # Check that we get correct number of action heads
        assert len(action_logits) == len(action_dims)

        # Check shapes of action logits
        assert action_logits[0].shape == (batch_size, action_dims[0])
        assert action_logits[1].shape == (batch_size, action_dims[1])

        # Check value shape
        assert value.shape == (batch_size, 1)

    def test_get_action_deterministic(self):
        """Test deterministic action selection returns argmax."""
        obs_dim = 42
        action_dims = [10, 2]

        policy = PolicyNetwork(obs_dim=obs_dim, action_dims=action_dims, hidden_size=64)
        obs = torch.randn(1, obs_dim)

        actions, log_prob, value = policy.get_action(obs, deterministic=True)

        # Check number of actions
        assert len(actions) == len(action_dims)

        # In deterministic mode, log_prob should be None
        assert log_prob is None

        # Check value shape
        assert value.shape == (1, 1)

    def test_get_action_stochastic(self):
        """Test stochastic action sampling returns log probabilities."""
        obs_dim = 42
        action_dims = [10, 2]

        policy = PolicyNetwork(obs_dim=obs_dim, action_dims=action_dims, hidden_size=64)
        obs = torch.randn(1, obs_dim)

        actions, log_prob, value = policy.get_action(obs, deterministic=False)

        # Check number of actions
        assert len(actions) == len(action_dims)

        # In stochastic mode, log_prob should exist
        assert log_prob is not None
        assert log_prob.shape == (1,)

        # Log probabilities should be negative (log of probability < 1)
        assert (log_prob <= 0).all()

    def test_checkpoint_save_load(self):
        """Test that checkpoint saving and loading preserves network behavior."""
        obs_dim = 42
        action_dims = [10, 2]
        hidden_size = 64

        # Create first policy
        policy1 = PolicyNetwork(obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size)
        obs = torch.randn(1, obs_dim)

        # Save checkpoint
        checkpoint = {
            'policy_state_dict': policy1.state_dict(),
            'obs_dim': obs_dim,
            'action_dims': action_dims,
            'hidden_size': hidden_size,
        }

        # Create second policy and load checkpoint
        policy2 = PolicyNetwork(
            obs_dim=checkpoint['obs_dim'],
            action_dims=checkpoint['action_dims'],
            hidden_size=checkpoint['hidden_size']
        )
        policy2.load_state_dict(checkpoint['policy_state_dict'])

        # Verify identical outputs
        with torch.no_grad():
            policy1.eval()
            policy2.eval()

            logits1, value1 = policy1(obs)
            logits2, value2 = policy2(obs)

            # Check action logits match
            for i in range(len(action_dims)):
                assert torch.allclose(logits1[i], logits2[i], atol=1e-6)

            # Check values match
            assert torch.allclose(value1, value2, atol=1e-6)
