import unittest

import jax.numpy as jnp
import numpy as np

from curemix.main_algo.reward_normalization import compute_forward_returns, init_reward_normalization_stats


class TestRewardNormalization(unittest.TestCase):
    def test_done_aware_forward_returns_reset_using_previous_done(self):
        running_forward_return = jnp.asarray([[10.0]], dtype=jnp.float32)  # [B=1, R=1]
        previous_done = jnp.asarray([[True]], dtype=jnp.bool_)  # [B=1, R=1]
        rewards = jnp.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=jnp.float32)  # [T=3, B=1, R=1]
        done = jnp.asarray([[[False]], [[True]], [[False]]], dtype=jnp.bool_)  # [T=3, B=1, R=1]
        gamma = jnp.asarray([0.9], dtype=jnp.float32)  # [R=1]

        new_running, new_previous_done, forward_returns = compute_forward_returns(
            running_forward_return=running_forward_return,
            previous_done=previous_done,
            rewards=rewards,
            done=done,
            gamma=gamma,
        )

        expected_forward_returns = np.asarray([[[1.0]], [[2.9]], [[3.0]]], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(forward_returns), expected_forward_returns, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(new_running), np.asarray([[3.0]], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(new_previous_done), np.asarray([[False]], dtype=bool))

    def test_non_episodic_forward_returns_match_continuous_accumulation(self):
        running_forward_return = jnp.asarray([[4.0]], dtype=jnp.float32)  # [B=1, R=1]
        previous_done = jnp.asarray([[False]], dtype=jnp.bool_)
        rewards = jnp.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=jnp.float32)
        done = jnp.zeros_like(rewards, dtype=jnp.bool_)
        gamma = jnp.asarray([0.5], dtype=jnp.float32)

        new_running, new_previous_done, forward_returns = compute_forward_returns(
            running_forward_return=running_forward_return,
            previous_done=previous_done,
            rewards=rewards,
            done=done,
            gamma=gamma,
        )

        expected_forward_returns = np.asarray([[[3.0]], [[3.5]], [[4.75]]], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(forward_returns), expected_forward_returns, rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(new_running), np.asarray([[4.75]], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(new_previous_done), np.asarray([[False]], dtype=bool))

    def test_init_stats_contains_previous_done_state(self):
        stats = init_reward_normalization_stats(num_envs=3, num_reward_functions=2)
        self.assertEqual(stats.running_forward_return.shape, (3, 2))
        self.assertEqual(stats.previous_done.shape, (3, 2))
        np.testing.assert_array_equal(np.asarray(stats.previous_done), np.zeros((3, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
