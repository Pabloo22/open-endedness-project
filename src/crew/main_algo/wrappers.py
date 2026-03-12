# copied from https://github.com/MichaelTMatthews/Craftax_Baselines/blob/150fd6ce26f77c37ebbd5087563f37f80f905727/wrappers.py#L83
# We added the `SparseCraftaxWrapper` class

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from craftax.craftax import constants as craftax_constants
from craftax.craftax.envs.craftax_symbolic_env import EnvState
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnv,
    CraftaxClassicSymbolicEnvNoAutoReset,
    EnvState as ClassicEnvState,
)
from craftax.craftax_classic import constants as classic_craftax_constants


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)
        rng, _rng = jax.random.split(rng)
        reset_priority = jnp.where(
            done,
            jax.random.uniform(_rng, (self.num_envs,), dtype=jnp.float32),
            -jnp.inf,
        )
        _, prioritized_done_envs = jax.lax.top_k(reset_priority, self.num_resets)
        prioritized_done_mask = reset_priority[prioritized_done_envs] > -jnp.inf

        def assign_unique_reset(slot_idx, current_reset_indexes):
            env_idx = prioritized_done_envs[slot_idx]

            def _set_index(indexes):
                return indexes.at[env_idx].set(jnp.asarray(slot_idx, dtype=indexes.dtype))

            return jax.lax.cond(
                prioritized_done_mask[slot_idx],
                _set_index,
                lambda indexes: indexes,
                current_reset_indexes,
            )

        reset_indexes = jax.lax.fori_loop(0, self.num_resets, assign_unique_reset, reset_indexes)

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class SparseCraftaxWrapper(GymnaxWrapper):
    """Wraps any Craftax environment (symbolic or pixel) to sparsify extrinsic
    rewards by removing the contribution of specified achievements.

    Args:
        env: The Craftax environment to wrap.
        blocked_achievement_ids:
            Optional list of ``Achievement`` integer values whose reward
            contribution is removed. Defaults to not block any achievements.
            The original dense reward is stored in ``info["real_reward"]``.
    """

    def __init__(
        self,
        env,
        blocked_achievement_ids: Sequence[int] | None = None,
        remove_health_reward: bool = False,
    ):
        super().__init__(env)

        if isinstance(env, (CraftaxClassicSymbolicEnv, CraftaxClassicSymbolicEnvNoAutoReset)):
            num_achievements = len(classic_craftax_constants.Achievement)
            # Classic craftax doesn't have a reward map because its rewards are all 1, so we can just create a mask.
            achievement_reward_map = jnp.ones(num_achievements, dtype=jnp.float32)
        else:
            num_achievements = len(craftax_constants.Achievement)
            achievement_reward_map = craftax_constants.ACHIEVEMENT_REWARD_MAP

        if blocked_achievement_ids is None:
            blocked_achievement_ids = []
        blocked_mask = jnp.zeros(num_achievements, dtype=jnp.bool_)
        self._blocked_mask = blocked_mask.at[jnp.array(blocked_achievement_ids, dtype=jnp.int32)].set(True)
        self._blocked_reward_map: jnp.ndarray = (
            jnp.array(achievement_reward_map, dtype=jnp.float32) * self._blocked_mask
        )
        self.remove_health_rewards = remove_health_reward

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state: EnvState | ClassicEnvState, action, params=None):
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)
        newly_unlocked = next_state.achievements & ~state.achievements
        blocked_reward = (newly_unlocked.astype(jnp.float32) * self._blocked_reward_map).sum()
        info["real_reward"] = reward
        adjusted_reward = reward - blocked_reward
        if self.remove_health_rewards:
            init_health = state.player_health
            adjusted_reward -= (next_state.player_health - init_health) * 0.1
        return obs, next_state, adjusted_reward, done, info
