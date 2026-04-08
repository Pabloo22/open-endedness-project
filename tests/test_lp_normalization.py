import unittest

import jax.numpy as jnp
import numpy as np

from curemix.main_algo.curriculum.lp_normalization import (
    compute_lp_batch_moments,
    init_lp_normalization_stats,
    update_lp_normalization_stats,
    update_lp_normalization_stats_from_data,
)
from curemix.main_algo.types import LpEstimationData


class TestLpNormalization(unittest.TestCase):
    def test_episodic_batch_moments_use_equal_environment_weighting(self):
        # Shapes:
        # - raw_rewards, done_masks: [U=1, T=6, B=2, R=1]
        raw_rewards = np.zeros((1, 6, 2, 1), dtype=np.float32)
        done_masks = np.zeros((1, 6, 2, 1), dtype=bool)

        # Env 0: 6 short episodes, each return=1.
        raw_rewards[0, :, 0, 0] = 1.0
        done_masks[0, :, 0, 0] = True

        # Env 1: one long episode, return=6.
        raw_rewards[0, 0, 1, 0] = 6.0
        done_masks[0, 5, 1, 0] = True

        lp_data = LpEstimationData(
            raw_rewards=jnp.asarray(raw_rewards),
            done_masks=jnp.asarray(done_masks),
        )

        mean_per_reward, second_moment_per_reward, valid_mask_per_reward = compute_lp_batch_moments(
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=jnp.asarray([True], dtype=jnp.bool_),
            gamma_per_reward_function=jnp.asarray([0.99], dtype=jnp.float32),
            eps=1e-8,
        )

        np.testing.assert_allclose(np.asarray(mean_per_reward), np.asarray([3.5], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(second_moment_per_reward),
            np.asarray([18.5], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(np.asarray(valid_mask_per_reward), np.asarray([True], dtype=bool))

    def test_continuous_batch_moments_use_mean_reward_initialization(self):
        # Rewards for one non-episodic stream, gamma=0.5.
        raw_rewards = np.asarray([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32)  # [1, 3, 1, 1]
        done_masks = np.zeros_like(raw_rewards, dtype=bool)
        lp_data = LpEstimationData(
            raw_rewards=jnp.asarray(raw_rewards),
            done_masks=jnp.asarray(done_masks),
        )

        mean_per_reward, second_moment_per_reward, valid_mask_per_reward = compute_lp_batch_moments(
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=jnp.asarray([False], dtype=jnp.bool_),
            gamma_per_reward_function=jnp.asarray([0.5], dtype=jnp.float32),
            eps=1e-8,
        )

        np.testing.assert_allclose(np.asarray(mean_per_reward), np.asarray([3.75], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(second_moment_per_reward),
            np.asarray([14.604166], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(np.asarray(valid_mask_per_reward), np.asarray([True], dtype=bool))

    def test_update_stats_bootstrap_then_ema(self):
        stats = init_lp_normalization_stats(num_reward_functions=1)
        stats = update_lp_normalization_stats(
            old_stats=stats,
            batch_mean_per_reward=jnp.asarray([2.0], dtype=jnp.float32),
            batch_second_moment_per_reward=jnp.asarray([10.0], dtype=jnp.float32),
            valid_mask_per_reward=jnp.asarray([True], dtype=jnp.bool_),
            ema_beta=0.05,
        )
        np.testing.assert_allclose(np.asarray(stats.mean), np.asarray([2.0], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(stats.second_moment), np.asarray([10.0], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(stats.var), np.asarray([6.0], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(stats.initialized), np.asarray([True], dtype=bool))

        stats = update_lp_normalization_stats(
            old_stats=stats,
            batch_mean_per_reward=jnp.asarray([4.0], dtype=jnp.float32),
            batch_second_moment_per_reward=jnp.asarray([20.0], dtype=jnp.float32),
            valid_mask_per_reward=jnp.asarray([True], dtype=jnp.bool_),
            ema_beta=0.05,
        )
        np.testing.assert_allclose(np.asarray(stats.mean), np.asarray([2.1], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(stats.second_moment), np.asarray([10.5], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(stats.var), np.asarray([6.09], dtype=np.float32), rtol=0, atol=1e-6)

        std = jnp.sqrt(stats.var + 1e-8)
        np.testing.assert_allclose(np.asarray(std), np.sqrt(np.asarray([6.09], dtype=np.float32)), rtol=0, atol=1e-6)

    def test_update_from_data_keeps_stats_when_episodic_stream_has_no_completed_episodes(self):
        stats = init_lp_normalization_stats(num_reward_functions=1)
        stats = update_lp_normalization_stats(
            old_stats=stats,
            batch_mean_per_reward=jnp.asarray([1.5], dtype=jnp.float32),
            batch_second_moment_per_reward=jnp.asarray([5.0], dtype=jnp.float32),
            valid_mask_per_reward=jnp.asarray([True], dtype=jnp.bool_),
            ema_beta=0.05,
        )

        raw_rewards = np.zeros((1, 4, 2, 1), dtype=np.float32)
        done_masks = np.zeros((1, 4, 2, 1), dtype=bool)  # No completed episodes.
        lp_data = LpEstimationData(
            raw_rewards=jnp.asarray(raw_rewards),
            done_masks=jnp.asarray(done_masks),
        )
        new_stats = update_lp_normalization_stats_from_data(
            old_stats=stats,
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=jnp.asarray([True], dtype=jnp.bool_),
            gamma_per_reward_function=jnp.asarray([0.99], dtype=jnp.float32),
            ema_beta=0.05,
            eps=1e-8,
        )

        np.testing.assert_allclose(np.asarray(new_stats.mean), np.asarray(stats.mean), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(new_stats.second_moment),
            np.asarray(stats.second_moment),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(np.asarray(new_stats.var), np.asarray(stats.var), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(new_stats.initialized), np.asarray(stats.initialized))


if __name__ == "__main__":
    unittest.main()
