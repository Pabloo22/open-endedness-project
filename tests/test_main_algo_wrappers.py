import unittest

import jax
import jax.numpy as jnp
import numpy as np

from crew.main_algo.setups import _resolve_optimistic_reset_ratio
from crew.main_algo.wrappers import OptimisticResetVecEnvWrapper


class _DummyVecEnv:
    def reset(self, key, params=None):
        del key, params
        state = jnp.array(-5, dtype=jnp.int32)
        obs = jnp.asarray([state.astype(jnp.float32)], dtype=jnp.float32)
        return obs, state

    def step(self, key, state, action, params=None):
        del key, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        done = action.astype(jnp.bool_)
        reward = action.astype(jnp.float32)
        obs = jnp.asarray([next_state.astype(jnp.float32)], dtype=jnp.float32)
        return obs, next_state, reward, done, {}


class TestOptimisticResetVecEnvWrapper(unittest.TestCase):
    def test_step_with_no_done_envs_keeps_all_states(self):
        env = OptimisticResetVecEnvWrapper(_DummyVecEnv(), num_envs=4, reset_ratio=2)

        obs, state, reward, done, _ = env.step(
            jax.random.key(0),
            jnp.arange(4, dtype=jnp.int32),
            jnp.zeros((4,), dtype=jnp.int32),
            None,
        )

        np.testing.assert_array_equal(np.asarray(done), np.zeros((4,), dtype=bool))
        np.testing.assert_array_equal(np.asarray(reward), np.zeros((4,), dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(state), np.arange(1, 5, dtype=np.int32))
        np.testing.assert_allclose(np.asarray(obs[:, 0]), np.arange(1, 5, dtype=np.float32), rtol=0, atol=0)

    def test_step_with_fewer_done_envs_than_num_resets_only_resets_done_envs(self):
        env = OptimisticResetVecEnvWrapper(_DummyVecEnv(), num_envs=4, reset_ratio=2)

        obs, state, reward, done, _ = env.step(
            jax.random.key(1),
            jnp.arange(4, dtype=jnp.int32),
            jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
            None,
        )

        np.testing.assert_array_equal(np.asarray(done), np.asarray([False, True, False, False]))
        np.testing.assert_array_equal(np.asarray(reward), np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(state), np.asarray([1, -5, 3, 4], dtype=np.int32))
        np.testing.assert_allclose(np.asarray(obs[:, 0]), np.asarray([1.0, -5.0, 3.0, 4.0], dtype=np.float32), rtol=0, atol=0)


class TestOptimisticResetRatioResolution(unittest.TestCase):
    def test_resolves_largest_divisor_within_limit(self):
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=18, ratio_limit=16), 9)
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=12, ratio_limit=16), 12)
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=7, ratio_limit=16), 7)
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=20, ratio_limit=6), 5)


if __name__ == "__main__":
    unittest.main()