# copied from https://github.com/MichaelTMatthews/Craftax_Baselines/blob/150fd6ce26f77c37ebbd5087563f37f80f905727/wrappers.py#L83
# We added the `SparseCraftaxWrapper` class

from functools import partial

import jax
import jax.numpy as jnp
from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement


BASIC_ACHIEVEMENT_IDS = list(range(25))


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

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

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
            contribution is removed. Defaults to blocking the basic achievements
            (IDs 0-24). The original dense reward is stored in ``info["real_reward"]``.
    """

    def __init__(self, env, blocked_achievement_ids: list[int] | None = None):
        super().__init__(env)

        num_achievements = len(Achievement)
        if blocked_achievement_ids is None:
            blocked_achievement_ids = BASIC_ACHIEVEMENT_IDS
        blocked_mask = jnp.zeros(num_achievements, dtype=jnp.bool_)
        self._blocked_mask = blocked_mask.at[jnp.array(blocked_achievement_ids, dtype=jnp.int32)].set(True)
        self._blocked_reward_map: jnp.ndarray = (
            jnp.array(ACHIEVEMENT_REWARD_MAP, dtype=jnp.float32) * self._blocked_mask
        )

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)

        newly_unlocked = next_state.achievements & ~state.achievements
        blocked_reward = (newly_unlocked.astype(jnp.float32) * self._blocked_reward_map).sum()
        info["real_reward"] = reward
        adjusted_reward = reward - blocked_reward

        return obs, next_state, adjusted_reward, done, info
