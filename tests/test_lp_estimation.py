import unittest

import jax
import jax.numpy as jnp
import numpy as np

from curemix.main_algo.curriculum.lp_estimation import estimate_lp_per_reward_function
from curemix.main_algo.types import LpEstimationData


def _ols_slope_numpy(x: np.ndarray, y: np.ndarray, eps: float) -> float:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = np.sum(x_centered**2)
    numerator = np.sum(x_centered * y_centered)
    if denominator <= eps:
        return 0.0
    return float(numerator / denominator)


def _wls_slope_numpy(x: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float) -> float:
    total_weight = np.sum(w)
    if total_weight <= eps:
        return 0.0

    x_weighted_mean = float(np.sum(w * x) / total_weight)
    y_weighted_mean = float(np.sum(w * y) / total_weight)
    x_centered = x - x_weighted_mean
    y_centered = y - y_weighted_mean
    denominator = np.sum(w * (x_centered**2))
    numerator = np.sum(w * x_centered * y_centered)
    if denominator <= eps:
        return 0.0
    return float(numerator / denominator)


def _debug_lifetime_intermediates(rewards_ut: np.ndarray, gamma: float, eps: float) -> dict[str, np.ndarray | float]:
    mean_reward_per_update = rewards_ut.mean(axis=1)
    discount_denom = max(1.0 - gamma, eps)
    return_proxy_per_update = mean_reward_per_update / discount_denom
    x = np.arange(rewards_ut.shape[0], dtype=np.float64)
    raw_slope = _ols_slope_numpy(x=x, y=return_proxy_per_update, eps=eps)
    return {
        "mean_reward_per_update": mean_reward_per_update,
        "return_proxy_per_update": return_proxy_per_update,
        "raw_slope": raw_slope,
    }


def _debug_episodic_intermediates(
    rewards_ut: np.ndarray,
    done_ut: np.ndarray,
    eps: float,
) -> dict[str, np.ndarray | float]:
    num_updates, num_steps = rewards_ut.shape

    episode_return = 0.0
    steps_per_update = np.zeros((num_updates,), dtype=np.int32)

    done_flat_indices: list[int] = []
    done_episode_x: list[float] = []
    done_episode_y: list[float] = []
    done_episode_is_intra: list[bool] = []
    done_episode_intra_idx: list[int] = []

    flat_idx = 0
    for update_idx in range(num_updates):
        for step_idx in range(num_steps):
            reward_t = float(rewards_ut[update_idx, step_idx])
            done_t = bool(done_ut[update_idx, step_idx])

            episode_return += reward_t
            steps_per_update[update_idx] += 1

            if done_t:
                total_steps = int(np.sum(steps_per_update))
                fractions = steps_per_update.astype(np.float64) / max(total_steps, 1)
                episode_x = float(np.sum(fractions * np.arange(num_updates, dtype=np.float64)))

                touched_updates = int(np.sum(steps_per_update > 0))
                is_intra = touched_updates == 1
                intra_idx = int(np.argmax(steps_per_update))

                done_flat_indices.append(flat_idx)
                done_episode_x.append(episode_x)
                done_episode_y.append(episode_return)
                done_episode_is_intra.append(is_intra)
                done_episode_intra_idx.append(intra_idx)

                episode_return = 0.0
                steps_per_update = np.zeros((num_updates,), dtype=np.int32)

            flat_idx += 1

    # Aggregate intra-batch finished episodes by update index.
    intra_counts = np.zeros((num_updates,), dtype=np.float64)
    intra_sums = np.zeros((num_updates,), dtype=np.float64)
    multi_x: list[float] = []
    multi_y: list[float] = []

    for x, y, is_intra, intra_idx in zip(done_episode_x, done_episode_y, done_episode_is_intra, done_episode_intra_idx, strict=True):
        if is_intra:
            intra_counts[intra_idx] += 1.0
            intra_sums[intra_idx] += y
        else:
            multi_x.append(x)
            multi_y.append(y)

    intra_mean = np.divide(
        intra_sums,
        np.maximum(intra_counts, 1.0),
    )
    intra_valid = intra_counts > 0.0
    intra_weights = np.sqrt(intra_counts)

    x_points = np.concatenate(
        [
            np.arange(num_updates, dtype=np.float64),
            np.asarray(multi_x, dtype=np.float64),
        ]
    )
    y_points = np.concatenate(
        [
            intra_mean.astype(np.float64),
            np.asarray(multi_y, dtype=np.float64),
        ]
    )
    w_points = np.concatenate(
        [
            intra_weights.astype(np.float64),
            np.ones((len(multi_x),), dtype=np.float64),
        ]
    )
    valid_points = np.concatenate(
        [
            intra_valid,
            np.ones((len(multi_x),), dtype=bool),
        ]
    )

    raw_slope = _wls_slope_numpy(
        x=x_points[valid_points],
        y=y_points[valid_points],
        w=w_points[valid_points],
        eps=eps,
    )

    return {
        "done_flat_indices": np.asarray(done_flat_indices, dtype=np.int32),
        "done_episode_x": np.asarray(done_episode_x, dtype=np.float64),
        "done_episode_y": np.asarray(done_episode_y, dtype=np.float64),
        "done_episode_is_intra": np.asarray(done_episode_is_intra, dtype=bool),
        "intra_counts": intra_counts,
        "intra_mean": intra_mean,
        "multi_x": np.asarray(multi_x, dtype=np.float64),
        "multi_y": np.asarray(multi_y, dtype=np.float64),
        "raw_slope": raw_slope,
    }


class TestLpAndScoreEstimation(unittest.TestCase):
    def test_lifetime_intermediates_and_final_lp(self):
        eps = 1e-8
        rewards_ut = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ],
            dtype=np.float64,
        )
        gamma = 0.5
        intermediates = _debug_lifetime_intermediates(rewards_ut=rewards_ut, gamma=gamma, eps=eps)

        np.testing.assert_allclose(intermediates["mean_reward_per_update"], np.asarray([1.0, 2.0, 3.0, 4.0]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(intermediates["return_proxy_per_update"], np.asarray([2.0, 4.0, 6.0, 8.0]), rtol=0, atol=1e-9)
        self.assertAlmostEqual(float(intermediates["raw_slope"]), 2.0, places=7)

        raw_rewards = jnp.asarray(rewards_ut[:, :, None, None], dtype=jnp.float32)  # [U, T, B=1, R=1]
        done_masks = jnp.zeros_like(raw_rewards, dtype=jnp.bool_)
        lp_data = LpEstimationData(raw_rewards=raw_rewards, done_masks=done_masks)

        lp, insufficient_episodes_env_mask = estimate_lp_per_reward_function(
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=jnp.asarray([False], dtype=jnp.bool_),
            gamma_per_reward_function=jnp.asarray([gamma], dtype=jnp.float32),
            eps=eps,
        )
        np.testing.assert_allclose(np.asarray(lp), np.asarray([[2.0]], dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(insufficient_episodes_env_mask), np.asarray([1], dtype=np.int32))

    def test_episodic_intermediates_and_final_lp(self):
        eps = 1e-8
        rewards_ut = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float64,
        )
        done_ut = np.asarray(
            [
                [False, True, False, False],
                [True, False, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )
        intermediates = _debug_episodic_intermediates(rewards_ut=rewards_ut, done_ut=done_ut, eps=eps)

        np.testing.assert_array_equal(intermediates["done_flat_indices"], np.asarray([1, 4, 6, 9, 10], dtype=np.int32))
        np.testing.assert_allclose(
            intermediates["done_episode_x"],
            np.asarray([0.0, 1.0 / 3.0, 1.0, 5.0 / 3.0, 2.0], dtype=np.float64),
            rtol=0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            intermediates["done_episode_y"],
            np.asarray([3.0, 12.0, 13.0, 27.0, 11.0], dtype=np.float64),
            rtol=0,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            intermediates["done_episode_is_intra"],
            np.asarray([True, False, True, False, True], dtype=bool),
        )
        np.testing.assert_allclose(intermediates["intra_counts"], np.asarray([1.0, 1.0, 1.0]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(intermediates["intra_mean"], np.asarray([3.0, 13.0, 11.0]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(intermediates["multi_x"], np.asarray([1.0 / 3.0, 5.0 / 3.0]), rtol=0, atol=1e-9)
        np.testing.assert_allclose(intermediates["multi_y"], np.asarray([12.0, 27.0]), rtol=0, atol=1e-9)
        self.assertAlmostEqual(float(intermediates["raw_slope"]), 6.23076923076923, places=7)

        raw_rewards = jnp.asarray(rewards_ut[:, :, None, None], dtype=jnp.float32)  # [U, T, B=1, R=1]
        done_masks = jnp.asarray(done_ut[:, :, None, None], dtype=jnp.bool_)
        lp_data = LpEstimationData(raw_rewards=raw_rewards, done_masks=done_masks)

        lp, insufficient_episodes_env_mask = estimate_lp_per_reward_function(
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=jnp.asarray([True], dtype=jnp.bool_),
            gamma_per_reward_function=jnp.asarray([0.99], dtype=jnp.float32),
            eps=eps,
        )
        expected_lp = np.asarray([[6.23076923076923]], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(lp), expected_lp, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(insufficient_episodes_env_mask), np.asarray([0], dtype=np.int32))

    def test_mixed_reward_types_dispatch_and_jit(self):
        eps = 1e-8
        U, T, B, R = 4, 3, 2, 2

        raw_rewards = np.zeros((U, T, B, R), dtype=np.float32)
        done_masks = np.zeros((U, T, B, R), dtype=bool)

        # Reward 0 (episodic): done every step; slope tracks per-update means.
        # Env 0: increasing means -> positive slope.
        # Env 1: decreasing means -> negative slope.
        for update_idx, mean_val in enumerate([1.0, 2.0, 3.0, 4.0]):
            raw_rewards[update_idx, :, 0, 0] = mean_val
        for update_idx, mean_val in enumerate([4.0, 3.0, 2.0, 1.0]):
            raw_rewards[update_idx, :, 1, 0] = mean_val
        done_masks[:, :, :, 0] = True

        # Reward 1 (lifetime): done mask ignored; use all-false mask.
        # Env 0: constant means -> zero slope.
        # Env 1: increasing means; with gamma=0.5 this yields slope=2 before normalization.
        for update_idx, mean_val in enumerate([1.0, 1.0, 1.0, 1.0]):
            raw_rewards[update_idx, :, 0, 1] = mean_val
        for update_idx, mean_val in enumerate([1.0, 2.0, 3.0, 4.0]):
            raw_rewards[update_idx, :, 1, 1] = mean_val

        lp_data = LpEstimationData(
            raw_rewards=jnp.asarray(raw_rewards),
            done_masks=jnp.asarray(done_masks),
        )
        is_episodic = jnp.asarray([True, False], dtype=jnp.bool_)
        gamma = jnp.asarray([0.99, 0.5], dtype=jnp.float32)

        lp, insufficient_episodes_env_mask = estimate_lp_per_reward_function(
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=is_episodic,
            gamma_per_reward_function=gamma,
            eps=eps,
        )
        expected = np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 2.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(lp), expected, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(insufficient_episodes_env_mask), np.asarray([0, 0], dtype=np.int32))

        lp_jit = jax.jit(estimate_lp_per_reward_function)(
            lp_data,
            is_episodic,
            gamma,
            eps,
        )
        lp_jit, insufficient_episodes_env_mask_jit = lp_jit
        np.testing.assert_allclose(np.asarray(lp_jit), expected, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(insufficient_episodes_env_mask_jit), np.asarray([0, 0], dtype=np.int32))

    def test_insufficient_episodes_mask_marks_envs_with_fewer_than_two_episodes(self):
        eps = 1e-8
        U, T, B, R = 3, 4, 3, 2

        raw_rewards = np.zeros((U, T, B, R), dtype=np.float32)
        done_masks = np.zeros((U, T, B, R), dtype=bool)

        # Env 0 has zero terminal events on extrinsic stream.
        # Env 1 has one terminal event on extrinsic stream.
        done_masks[1, 2, 1, 0] = True
        # Env 2 has two terminal events on extrinsic stream.
        done_masks[0, 1, 2, 0] = True
        done_masks[2, 3, 2, 0] = True

        lp_data = LpEstimationData(
            raw_rewards=jnp.asarray(raw_rewards),
            done_masks=jnp.asarray(done_masks),
        )
        is_episodic = jnp.asarray([True, False], dtype=jnp.bool_)
        gamma = jnp.asarray([0.99, 0.99], dtype=jnp.float32)

        _, insufficient_episodes_env_mask = estimate_lp_per_reward_function(
            lp_estimation_data=lp_data,
            is_episodic_per_reward_function=is_episodic,
            gamma_per_reward_function=gamma,
            eps=eps,
        )
        np.testing.assert_array_equal(np.asarray(insufficient_episodes_env_mask), np.asarray([1, 1, 0], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
