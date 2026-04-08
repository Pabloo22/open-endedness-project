import unittest

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from curemix.main_algo.setups import _resolve_optimistic_reset_ratio
from curemix.main_algo.wrappers import (
    FixedResetKeyEnvWrapper,
    OptimisticResetVecEnvWrapper,
    SparseCraftaxWrapper,
)


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


class _KeyedResetEnv:
    def reset(self, key, params=None):
        del params
        state = jnp.sum(jax.random.key_data(key).astype(jnp.int32))
        obs = jnp.asarray([state.astype(jnp.float32)], dtype=jnp.float32)
        return obs, state

    def step(self, key, state, action, params=None):
        del key, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        done = action.astype(jnp.bool_)
        reward = action.astype(jnp.float32)
        obs = jnp.asarray([next_state.astype(jnp.float32)], dtype=jnp.float32)
        return obs, next_state, reward, done, {}


def _state_from_seed(seed: int) -> np.int32:
    return np.int32(np.asarray(jax.random.key_data(jax.random.key(seed)), dtype=np.int32).sum())


class TestFixedResetKeyEnvWrapper(unittest.TestCase):
    def test_reset_ignores_incoming_keys(self):
        env = FixedResetKeyEnvWrapper(_KeyedResetEnv(), fixed_reset_seed=17)

        obs_first, state_first = env.reset(jax.random.key(1), None)
        obs_second, state_second = env.reset(jax.random.key(999), None)

        expected_state = _state_from_seed(17)
        np.testing.assert_array_equal(np.asarray(state_first), np.asarray(expected_state))
        np.testing.assert_array_equal(np.asarray(state_second), np.asarray(expected_state))
        np.testing.assert_allclose(
            np.asarray(obs_first),
            np.asarray([expected_state], dtype=np.float32),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(np.asarray(state_first), np.asarray(state_second))
        np.testing.assert_array_equal(np.asarray(obs_first), np.asarray(obs_second))


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

    def test_reset_uses_same_fixed_state_for_all_envs_when_wrapped(self):
        fixed_seed = 17
        env = OptimisticResetVecEnvWrapper(
            FixedResetKeyEnvWrapper(_KeyedResetEnv(), fixed_reset_seed=fixed_seed),
            num_envs=4,
            reset_ratio=2,
        )

        obs, state = env.reset(jax.random.key(0), None)

        expected_state = _state_from_seed(fixed_seed)
        np.testing.assert_array_equal(np.asarray(state), np.full((4,), expected_state, dtype=np.int32))
        np.testing.assert_allclose(
            np.asarray(obs[:, 0]),
            np.full((4,), expected_state, dtype=np.float32),
            rtol=0,
            atol=0,
        )

    def test_step_resets_done_envs_to_same_fixed_state_when_wrapped(self):
        fixed_seed = 17
        env = OptimisticResetVecEnvWrapper(
            FixedResetKeyEnvWrapper(_KeyedResetEnv(), fixed_reset_seed=fixed_seed),
            num_envs=4,
            reset_ratio=2,
        )

        obs, state, reward, done, _ = env.step(
            jax.random.key(1),
            jnp.arange(4, dtype=jnp.int32),
            jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
            None,
        )

        expected_state = _state_from_seed(fixed_seed)
        np.testing.assert_array_equal(np.asarray(done), np.asarray([False, True, False, False]))
        np.testing.assert_array_equal(np.asarray(reward), np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(state), np.asarray([1, expected_state, 3, 4], dtype=np.int32))
        np.testing.assert_allclose(
            np.asarray(obs[:, 0]),
            np.asarray([1.0, expected_state, 3.0, 4.0], dtype=np.float32),
            rtol=0,
            atol=0,
        )


class TestOptimisticResetRatioResolution(unittest.TestCase):
    def test_resolves_largest_divisor_within_limit(self):
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=18, ratio_limit=16), 9)
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=12, ratio_limit=16), 12)
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=7, ratio_limit=16), 7)
        self.assertEqual(_resolve_optimistic_reset_ratio(num_envs=20, ratio_limit=6), 5)


class TestFixedResetKeyEnvWrapperCraftaxIntegration(unittest.TestCase):
    def _make_craftax_env(self, fixed_reset_seed: int):
        base_env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
        env_params = base_env.default_params.replace(max_timesteps=32)
        base_env = SparseCraftaxWrapper(base_env)
        env = OptimisticResetVecEnvWrapper(
            FixedResetKeyEnvWrapper(base_env, fixed_reset_seed=fixed_reset_seed),
            num_envs=4,
            reset_ratio=2,
        )
        return env, env_params

    def test_craftax_resets_repeat_the_same_world_with_fixed_reset_wrapper(self):
        env, env_params = self._make_craftax_env(fixed_reset_seed=17)

        obs_first, state_first = env.reset(jax.random.key(0), env_params)
        obs_second, state_second = env.reset(jax.random.key(999), env_params)

        np.testing.assert_array_equal(np.asarray(obs_first), np.asarray(obs_second))
        np.testing.assert_array_equal(np.asarray(state_first.map), np.asarray(state_second.map))
        np.testing.assert_array_equal(
            np.asarray(state_first.player_position),
            np.asarray(state_second.player_position),
        )

        reference_map = np.asarray(state_first.map[0])
        reference_obs = np.asarray(obs_first[0])
        reference_player_position = np.asarray(state_first.player_position[0])
        for env_idx in range(1, 4):
            np.testing.assert_array_equal(reference_map, np.asarray(state_first.map[env_idx]))
            np.testing.assert_array_equal(reference_obs, np.asarray(obs_first[env_idx]))
            np.testing.assert_array_equal(
                reference_player_position,
                np.asarray(state_first.player_position[env_idx]),
            )


if __name__ == "__main__":
    unittest.main()
