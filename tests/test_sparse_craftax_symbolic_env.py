from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import ACHIEVEMENT_REWARD_MAP, Achievement
from crew.main_algo.sparse_craftax_symbolic_env import SparseCraftaxSymbolicEnv, BASIC_ACHIEVEMENT_IDS


def _make_achievements(true_ids: list[int]) -> jnp.ndarray:
    """Return a boolean achievement array with the given IDs set to True."""
    arr = jnp.zeros(len(Achievement), dtype=jnp.bool_)
    if true_ids:
        arr = arr.at[jnp.array(true_ids, dtype=jnp.int32)].set(True)
    return arr


def _fake_step_env(real_reward: float, next_achievements: jnp.ndarray):
    """Return a mock for CraftaxSymbolicEnv.step_env that yields controlled values."""
    next_state = SimpleNamespace(achievements=next_achievements)
    info = {f"Achievements/{a.name.lower()}": jnp.array(0.0) for a in Achievement}

    def _mock(unused_self, unused_rng, unused_state, unused_action, unused_params):
        return jnp.zeros(1), next_state, jnp.array(real_reward, dtype=jnp.float32), jnp.array(False), info

    return _mock


class TestAdjustReward(unittest.TestCase):
    def setUp(self):
        self.env = SparseCraftaxSymbolicEnv()

    def test_blocked_achievement_subtracts_its_reward(self):
        # Achievement 0 is blocked by default; find out its reward.
        blocked_id = BASIC_ACHIEVEMENT_IDS[0]
        expected_subtraction = float(ACHIEVEMENT_REWARD_MAP[blocked_id])

        newly_unlocked = _make_achievements([blocked_id])
        real_reward = 1.0
        adjusted = self.env._adjust_reward(jnp.array(real_reward), newly_unlocked)

        np.testing.assert_allclose(float(adjusted), real_reward - expected_subtraction, rtol=1e-6)

    def test_non_blocked_achievement_does_not_change_reward(self):
        # Find the first achievement NOT in the blocked list.
        non_blocked_id = next(i for i in range(len(Achievement)) if i not in BASIC_ACHIEVEMENT_IDS)

        newly_unlocked = _make_achievements([non_blocked_id])
        real_reward = 2.5
        adjusted = self.env._adjust_reward(jnp.array(real_reward), newly_unlocked)

        np.testing.assert_allclose(float(adjusted), real_reward, rtol=1e-6)

    def test_no_newly_unlocked_leaves_reward_unchanged(self):
        newly_unlocked = _make_achievements([])
        real_reward = 3.0
        adjusted = self.env._adjust_reward(jnp.array(real_reward), newly_unlocked)

        np.testing.assert_allclose(float(adjusted), real_reward, rtol=1e-6)

    def test_multiple_blocked_achievements_all_subtracted(self):
        ids = BASIC_ACHIEVEMENT_IDS[:3]
        expected_subtraction = sum(float(ACHIEVEMENT_REWARD_MAP[i]) for i in ids)

        newly_unlocked = _make_achievements(ids)
        real_reward = 10.0
        adjusted = self.env._adjust_reward(jnp.array(real_reward), newly_unlocked)

        np.testing.assert_allclose(float(adjusted), real_reward - expected_subtraction, rtol=1e-6)

    def test_adjust_reward_works_under_jit(self):
        """Regression: float() in _adjust_reward would raise ConcretizationTypeError under JIT."""
        blocked_id = BASIC_ACHIEVEMENT_IDS[0]
        newly_unlocked = _make_achievements([blocked_id])
        real_reward = jnp.array(1.0)

        jitted = jax.jit(self.env._adjust_reward)
        # Should not raise.
        result = jitted(real_reward, newly_unlocked)
        self.assertIsInstance(result, jnp.ndarray)

    def test_custom_blocked_ids_respected(self):
        # Block only a non-basic achievement (the first one outside BASIC_ACHIEVEMENT_IDS).
        custom_id = next(i for i in range(len(Achievement)) if i not in BASIC_ACHIEVEMENT_IDS)
        env = SparseCraftaxSymbolicEnv(blocked_achievement_ids=[custom_id])
        expected_subtraction = float(ACHIEVEMENT_REWARD_MAP[custom_id])

        newly_unlocked = _make_achievements([custom_id])
        adjusted = env._adjust_reward(jnp.array(5.0), newly_unlocked)

        np.testing.assert_allclose(float(adjusted), 5.0 - expected_subtraction, rtol=1e-6)


class TestStepEnv(unittest.TestCase):
    def setUp(self):
        self.env = SparseCraftaxSymbolicEnv()
        self.rng = jax.random.key(0)
        self.params = self.env.default_params

    def _run_step(self, prior_achievements: list[int], next_achievements: list[int], real_reward: float):
        prior_state = SimpleNamespace(achievements=_make_achievements(prior_achievements))
        mock = _fake_step_env(real_reward, _make_achievements(next_achievements))
        target = "craftax.craftax.envs.craftax_symbolic_env.CraftaxSymbolicEnv.step_env"
        with patch(target, mock):
            return self.env.step_env(self.rng, prior_state, action=0, params=self.params)

    def test_newly_unlocked_blocked_achievement_reduces_reward(self):
        blocked_id = BASIC_ACHIEVEMENT_IDS[0]
        real_reward = 1.0
        _, _, adjusted, _, _ = self._run_step([], [blocked_id], real_reward)
        expected = real_reward - float(ACHIEVEMENT_REWARD_MAP[blocked_id])
        np.testing.assert_allclose(float(adjusted), expected, rtol=1e-6)

    def test_already_unlocked_achievement_not_double_counted(self):
        blocked_id = BASIC_ACHIEVEMENT_IDS[0]
        real_reward = 1.0
        # Achievement was already unlocked before this step.
        _, _, adjusted, _, _ = self._run_step([blocked_id], [blocked_id], real_reward)
        # No new unlock → reward should be unchanged.
        np.testing.assert_allclose(float(adjusted), real_reward, rtol=1e-6)

    def test_real_reward_stored_in_info(self):
        real_reward = 7.0
        _, _, _, _, info = self._run_step([], [], real_reward)
        np.testing.assert_allclose(float(info["real_reward"]), real_reward, rtol=1e-6)

    def test_non_blocked_newly_unlocked_does_not_change_reward(self):
        non_blocked_id = next(i for i in range(len(Achievement)) if i not in BASIC_ACHIEVEMENT_IDS)
        real_reward = 4.0
        _, _, adjusted, _, _ = self._run_step([], [non_blocked_id], real_reward)
        np.testing.assert_allclose(float(adjusted), real_reward, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
