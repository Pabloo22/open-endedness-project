from __future__ import annotations

import chex
import jax.numpy as jnp

from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv, EnvState, EnvParams, StaticEnvParams


BASIC_ACHIEVEMENT_IDS = list(range(25))


class SparseCraftaxSymbolicEnv(CraftaxSymbolicEnv):  # pylint: disable=abstract-method
    """Craftax wrapper that sparsifies extrinsic rewards and combines them with
    an arbitrary number of intrinsic reward modules.

    Args:
        blocked_achievement_ids:
            Optional list of `Achievement` integer values whose reward
            contribution is removed before accumulation. All other
            reward signal (health reward, non-blocked achievements) is unchanged.
            If `None`, defaults to blocking the basic achievements (IDs 0-24).
    """

    def __init__(
        self,
        static_env_params: StaticEnvParams | None = None,
        blocked_achievement_ids: list[int] | None = None,
    ):
        super().__init__(static_env_params)
        num_achievements = len(Achievement)
        if blocked_achievement_ids is None:
            blocked_achievement_ids = BASIC_ACHIEVEMENT_IDS
        blocked_mask = jnp.zeros(num_achievements, dtype=jnp.bool_)
        self._blocked_mask = blocked_mask.at[jnp.array(blocked_achievement_ids, dtype=jnp.int32)].set(True)
        # Pre-compute the reward that each blocked achievement would contribute.
        self._blocked_reward_map: jnp.ndarray = (
            jnp.array(ACHIEVEMENT_REWARD_MAP, dtype=jnp.float32) * self._blocked_mask
        )

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        next_obs, next_env_state, real_reward, done, info = super().step_env(rng, state, action, params)

        # Reconstruct correct achievements and remove blocked reward.
        newly_unlocked = next_env_state.achievements & ~state.achievements
        adjusted_reward = self._adjust_reward(real_reward, newly_unlocked)
        info["real_reward"] = real_reward

        # Note: There is a bug in the current implementation of `CraftaxSymbolicEnv`, the real return type of
        # the reward is not `float` (as the type hint suggests) but a traced jax scalar.
        return next_obs, next_env_state, adjusted_reward, done, info  # type: ignore[return-value]

    def _adjust_reward(self, real_reward: float, newly_unlocked: jnp.ndarray) -> jnp.ndarray:
        """Subtract the reward contribution of blocked achievements."""
        blocked_reward_this_step = (newly_unlocked.astype(jnp.float32) * self._blocked_reward_map).sum()
        return real_reward - blocked_reward_this_step
