"""Tests for craftax_sparsified_simple_loop.py

Run with:
    pytest notebooks/test_craftax_sparsified_simple_loop.py -v
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from crew.scripts.craftax_sparsified_simple_loop import (
    Config,
    SparseIntrinsicCraftaxWrapper,
)

# Number of achievements in this version of Craftax.
_NUM_ACHIEVEMENTS = len(Achievement)


# ---------------------------------------------------------------------------
# Stub environment (no Craftax / GPU dependency)
# ---------------------------------------------------------------------------


class _StubEnvState:
    """Trivial state: step counter + boolean achievement flags."""

    def __init__(self, timestep, achievements=None):
        self.timestep = jnp.int32(timestep)
        if achievements is None:
            achievements = jnp.zeros(_NUM_ACHIEVEMENTS, dtype=jnp.bool_)
        self.achievements = achievements


# Register as a JAX pytree so it survives inside SparseWrapperState (a flax struct).
jax.tree_util.register_pytree_node(
    _StubEnvState,
    flatten_func=lambda s: ([s.timestep, s.achievements], None),
    unflatten_func=lambda _, vals: _StubEnvState(int(vals[0]), vals[1]),
)


class StubEnv:
    """Deterministic env: constant base reward each step, done after
    ``episode_length`` steps.  Optionally unlocks specific achievements on
    given step indices, adding their ``ACHIEVEMENT_REWARD_MAP`` reward to the
    step reward.

    Args:
        obs_dim: Dimensionality of observations.
        episode_length: Number of steps before ``done=True``.
        reward: Base reward returned on every step (before achievements).
        achievement_unlocks: Mapping of {step_index: [achievement_ids]} to
            unlock.  Achievement IDs must be valid indices into the
            ``Achievement`` enum.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        episode_length: int = 3,
        reward: float = 1.0,
        achievement_unlocks: dict[int, list[int]] | None = None,
    ):
        self._obs_dim = obs_dim
        self._episode_length = episode_length
        self._reward = reward
        self._achievement_unlocks: dict[int, list[int]] = (
            achievement_unlocks if achievement_unlocks is not None else {}
        )

    @property
    def default_params(self):
        return None

    def action_space(self, params):
        class _Space:
            n = 5

        return _Space()

    def observation_space(self, params):
        return None

    def reset(self, key, params=None):
        obs = jnp.zeros(self._obs_dim, dtype=jnp.float32)
        return obs, _StubEnvState(0)

    def step(self, key, state, action, params=None):
        obs = jnp.ones(self._obs_dim, dtype=jnp.float32)
        step_idx = int(state.timestep)
        done = jnp.bool_(state.timestep >= self._episode_length - 1)

        # Unlock achievements scheduled for this step.
        new_achievements = state.achievements
        for ach_id in self._achievement_unlocks.get(step_idx, []):
            new_achievements = new_achievements.at[ach_id].set(True)

        newly_unlocked = new_achievements & ~state.achievements
        achievement_reward = (
            newly_unlocked.astype(jnp.float32)
            * jnp.array(ACHIEVEMENT_REWARD_MAP, dtype=jnp.float32)
        ).sum()
        reward = jnp.float32(self._reward) + achievement_reward

        next_state = _StubEnvState(step_idx + 1, new_achievements)

        # Mimic log_achievements_to_info: values are non-zero only on done.
        info = {
            f"Achievements/{ach.name.lower()}": new_achievements[ach.value]
            * done.astype(jnp.float32)
            * 100.0
            for ach in Achievement
        }
        return obs, next_state, reward, done, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(
    episode_length: int = 3,
    reward_per_step: float = 1.0,
    alpha: list[float] | None = None,
    achievement_unlocks: dict[int, list[int]] | None = None,
    blocked_achievement_ids: list[int] | None = None,
) -> SparseIntrinsicCraftaxWrapper:
    env = StubEnv(
        episode_length=episode_length,
        reward=reward_per_step,
        achievement_unlocks=achievement_unlocks,
    )
    kwargs: dict = {}
    if alpha is not None:
        kwargs["alpha"] = alpha
    if blocked_achievement_ids is not None:
        kwargs["blocked_achievement_ids"] = blocked_achievement_ids
    return SparseIntrinsicCraftaxWrapper(env, modules=[], **kwargs)


def _run_episode(wrapper, episode_length: int):
    """Run exactly one episode and return (rewards, dones)."""
    rng = jax.random.PRNGKey(0)
    _, state = wrapper.reset(rng)
    rewards, dones = [], []
    for _ in range(episode_length):
        rng, step_rng = jax.random.split(rng)
        _, state, reward, done, _ = wrapper.step(step_rng, state, jnp.int32(0))
        rewards.append(float(reward))
        dones.append(bool(done))
    return rewards, dones, state


# ---------------------------------------------------------------------------
# Tests: SparseIntrinsicCraftaxWrapper — reward sparsification
# ---------------------------------------------------------------------------


class TestSparseReward:
    def test_reward_zero_on_non_terminal_steps(self):
        """Sparse reward must be 0 on every non-terminal step."""
        wrapper = _make_wrapper(episode_length=4)
        rewards, dones, _ = _run_episode(wrapper, episode_length=4)

        for step, (reward, done) in enumerate(zip(rewards, dones)):
            if not done:
                assert reward == pytest.approx(
                    0.0
                ), f"Expected reward=0 on non-terminal step {step}, got {reward}"

    def test_reward_equals_accumulated_sum_at_done(self):
        """Sparse reward at episode end should equal sum of all dense rewards."""
        episode_length = 5
        reward_per_step = 2.0
        wrapper = _make_wrapper(
            episode_length=episode_length, reward_per_step=reward_per_step
        )
        rewards, dones, _ = _run_episode(wrapper, episode_length)

        assert dones[-1], "Last step should be done=True"
        assert rewards[-1] == pytest.approx(episode_length * reward_per_step), (
            f"Expected total sparse reward {episode_length * reward_per_step}, "
            f"got {rewards[-1]}"
        )

    def test_episode_return_resets_after_done(self):
        """episode_return inside the state should reset to 0 after a terminal step."""
        wrapper = _make_wrapper(episode_length=3)
        _, _, final_state = _run_episode(wrapper, episode_length=3)

        assert float(final_state.episode_return) == pytest.approx(0.0)

    def test_prev_obs_matches_returned_obs(self):
        """state.prev_obs after a step should equal the observation returned by that
        step."""
        wrapper = _make_wrapper()
        rng = jax.random.PRNGKey(42)
        _, state = wrapper.reset(rng)

        rng, step_rng = jax.random.split(rng)
        obs, new_state, _, _, _ = wrapper.step(step_rng, state, jnp.int32(0))

        assert jnp.allclose(new_state.prev_obs, obs)

    def test_done_flag_raised_on_last_step(self):
        """done should be False for all steps except the last one."""
        episode_length = 3
        wrapper = _make_wrapper(episode_length=episode_length)
        _, dones, _ = _run_episode(wrapper, episode_length)

        assert dones == [False] * (episode_length - 1) + [True]

    def test_alpha_scales_terminal_reward(self):
        """alpha[0] should scale the sparse extrinsic reward."""
        alpha_scale = 0.5
        episode_length = 4
        reward_per_step = 1.0
        wrapper = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=reward_per_step,
            alpha=[alpha_scale],
        )
        rewards, dones, _ = _run_episode(wrapper, episode_length)

        expected = episode_length * reward_per_step * alpha_scale
        assert rewards[-1] == pytest.approx(expected)

    def test_alpha_length_mismatch_raises(self):
        """Passing alpha with wrong length should raise ValueError."""
        env = StubEnv()
        with pytest.raises(ValueError, match="len\\(alpha\\)"):
            SparseIntrinsicCraftaxWrapper(env, modules=[], alpha=[1.0, 0.5])

    def test_default_alpha_is_one_when_no_modules(self):
        """With no modules, the default alpha should leave rewards unscaled."""
        episode_length = 3
        reward_per_step = 2.0
        wrapper = _make_wrapper(
            episode_length=episode_length, reward_per_step=reward_per_step
        )
        rewards, _, _ = _run_episode(wrapper, episode_length)

        # Terminal reward should equal the full accumulated sum (alpha=1.0)
        assert rewards[-1] == pytest.approx(episode_length * reward_per_step)


# ---------------------------------------------------------------------------
# Tests: SparseIntrinsicCraftaxWrapper — blocked achievements
# ---------------------------------------------------------------------------

# Use the first two basic achievements (reward = 1.0 each per ACHIEVEMENT_REWARD_MAP).
_ACH_A = Achievement(0).value  # first achievement ID
_ACH_B = Achievement(1).value  # second achievement ID
_ACH_A_REWARD = float(ACHIEVEMENT_REWARD_MAP[_ACH_A])
_ACH_B_REWARD = float(ACHIEVEMENT_REWARD_MAP[_ACH_B])


class TestBlockedAchievements:
    def test_no_blocked_ids_leaves_reward_unchanged(self):
        """blocked_achievement_ids=None (default) must not alter rewards."""
        episode_length = 3
        base = 1.0
        # Achievement A fires on step 1.
        wrapper_default = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A]},
        )
        wrapper_explicit_none = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A]},
            blocked_achievement_ids=None,
        )
        r_default, _, _ = _run_episode(wrapper_default, episode_length)
        r_none, _, _ = _run_episode(wrapper_explicit_none, episode_length)
        assert r_default[-1] == pytest.approx(r_none[-1])

    def test_empty_blocked_ids_same_as_none(self):
        """blocked_achievement_ids=[] must be identical to None."""
        episode_length = 3
        base = 1.0
        wrapper_none = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A]},
        )
        wrapper_empty = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A]},
            blocked_achievement_ids=[],
        )
        r_none, _, _ = _run_episode(wrapper_none, episode_length)
        r_empty, _, _ = _run_episode(wrapper_empty, episode_length)
        assert r_none[-1] == pytest.approx(r_empty[-1])

    def test_blocked_achievement_removed_from_accumulated_reward(self):
        """Blocking an achievement should remove its reward from the episode total."""
        episode_length = 3
        base = 1.0
        # Achievement A fires on step 1.
        wrapper_unblocked = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A]},
        )
        wrapper_blocked = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A]},
            blocked_achievement_ids=[_ACH_A],
        )
        r_unblocked, _, _ = _run_episode(wrapper_unblocked, episode_length)
        r_blocked, _, _ = _run_episode(wrapper_blocked, episode_length)

        assert r_blocked[-1] == pytest.approx(
            r_unblocked[-1] - _ACH_A_REWARD
        ), "Terminal reward should be reduced by the blocked achievement's reward."

    def test_non_blocked_achievement_still_contributes(self):
        """Unblocked achievement rewards must remain intact."""
        episode_length = 3
        base = 0.0
        # Both A and B fire on step 1; only A is blocked.
        wrapper = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A, _ACH_B]},
            blocked_achievement_ids=[_ACH_A],
        )
        rewards, _, _ = _run_episode(wrapper, episode_length)

        # Only B's reward should survive.
        assert rewards[-1] == pytest.approx(_ACH_B_REWARD)

    def test_multiple_blocked_achievements_all_removed(self):
        """All blocked achievements must be removed; base reward is unaffected."""
        episode_length = 3
        base = 2.0
        wrapper = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=base,
            achievement_unlocks={1: [_ACH_A, _ACH_B]},
            blocked_achievement_ids=[_ACH_A, _ACH_B],
        )
        rewards, _, _ = _run_episode(wrapper, episode_length)

        assert rewards[-1] == pytest.approx(episode_length * base)

    def test_blocking_does_not_affect_non_terminal_steps(self):
        """Blocked achievements should never leak reward onto non-terminal steps."""
        episode_length = 4
        wrapper = _make_wrapper(
            episode_length=episode_length,
            reward_per_step=1.0,
            achievement_unlocks={1: [_ACH_A]},
            blocked_achievement_ids=[_ACH_A],
        )
        rewards, dones, _ = _run_episode(wrapper, episode_length)
        for step, (reward, done) in enumerate(zip(rewards, dones)):
            if not done:
                assert reward == pytest.approx(
                    0.0
                ), f"Expected 0 on non-terminal step {step}, got {reward}"


# ---------------------------------------------------------------------------
# Tests: Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_num_batches_rounds_up(self):
        """num_batches_of_envs should use ceiling division."""
        cfg = Config(
            total_timesteps=300_000, num_parallel_envs=64, num_steps_per_env=512
        )
        expected = math.ceil(300_000 / (64 * 512))
        assert cfg.num_batches_of_envs == expected

    def test_exact_division_gives_no_extra_batch(self):
        """When total_timesteps divides evenly, no extra batch is added."""
        cfg = Config(
            total_timesteps=64 * 512, num_parallel_envs=64, num_steps_per_env=512
        )
        assert cfg.num_batches_of_envs == 1

    def test_single_timestep_gives_one_batch(self):
        """Even total_timesteps=1 should result in at least one batch."""
        cfg = Config(total_timesteps=1, num_parallel_envs=128, num_steps_per_env=1024)
        assert cfg.num_batches_of_envs == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
