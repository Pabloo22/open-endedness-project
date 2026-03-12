import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax.tree_util import Partial

from crew.main_algo.actor_critic import Categorical
from crew.main_algo.evaluation import evaluate_policy_on_alphas, infer_achievement_names


class _DummyAutoResetEvalEnv:
    def reset(self, key, params=None):
        del key, params
        return jnp.array([0.0, 0.0], dtype=jnp.float32), jnp.array(0, dtype=jnp.int32)

    def step(self, key, state, action, params=None):
        del key, action, params
        next_state = state + jnp.array(1, dtype=jnp.int32)
        done = next_state >= 3

        obs_step = jnp.asarray([next_state.astype(jnp.float32), 0.0], dtype=jnp.float32)
        reset_obs = jnp.zeros((2,), dtype=jnp.float32)
        reward = jnp.where(done, jnp.array(1.0, dtype=jnp.float32), jnp.array(0.25, dtype=jnp.float32))

        info = {
            "Achievements/a": jnp.where(done, jnp.array(100.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
            "Achievements/b": jnp.array(0.0, dtype=jnp.float32),
            "score": reward,
        }

        obs_out = jax.lax.select(done, reset_obs, obs_step)
        state_out = jax.lax.select(done, jnp.array(0, dtype=jnp.int32), next_state)
        return obs_out, state_out, reward, done, info


def _dummy_apply_fn(params, memories, observations, mask, alpha_batch, method=None):
    del params, mask, method
    batch_size = observations.shape[0]
    num_reward_functions = alpha_batch.shape[1]
    logits = jnp.zeros((batch_size, 2), dtype=jnp.float32)
    pi = Categorical(logits=logits)
    values = jnp.zeros((batch_size, num_reward_functions), dtype=jnp.float32)
    memory_out = jnp.zeros((batch_size, memories.shape[2], memories.shape[3]), dtype=memories.dtype)
    return pi, values, memory_out


class TestMainAlgoEvaluation(unittest.TestCase):
    def _make_train_state(self) -> TrainState:
        return TrainState.create(
            apply_fn=_dummy_apply_fn,
            params={"w": jnp.array(0.0, dtype=jnp.float32)},
            tx=optax.sgd(1e-3),
        )

    def _make_config(self):
        return SimpleNamespace(
            past_context_length=4,
            num_transformer_blocks=1,
            transformer_hidden_states_dim=8,
            num_attn_heads=2,
        )

    def test_infer_achievement_names_returns_sorted_names(self):
        env = _DummyAutoResetEvalEnv()
        names = infer_achievement_names(env=env, env_params=None)
        self.assertEqual(names, ("Achievements/a", "Achievements/b"))

    def test_eval_outputs_shapes_and_terminal_achievement_logic(self):
        env = _DummyAutoResetEvalEnv()
        config = self._make_config()
        train_state = self._make_train_state()
        evaluation_alphas = jnp.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float32,
        )
        achievement_names = ("Achievements/a", "Achievements/b")

        _, eval_metrics = evaluate_policy_on_alphas(
            rng=jax.random.key(0),
            train_state=train_state,
            env=env,
            env_params=None,
            evaluation_alphas=evaluation_alphas,
            num_eval_envs=3,
            num_eval_episodes=4,
            achievement_names=achievement_names,
            config=config,
        )

        self.assertEqual(eval_metrics["eval/returns"].shape, (2, 3, 4))
        self.assertEqual(eval_metrics["eval/lengths"].shape, (2, 3, 4))
        self.assertEqual(eval_metrics["eval/achievements"].shape, (2, 3, 4, 2))
        np.testing.assert_allclose(np.asarray(eval_metrics["eval/lengths"]), np.full((2, 3, 4), 3, dtype=np.int32), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(eval_metrics["eval/returns"]), np.full((2, 3, 4), 1.5, dtype=np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(eval_metrics["eval/achievements"][..., 0]), np.ones((2, 3, 4), dtype=bool))
        np.testing.assert_array_equal(np.asarray(eval_metrics["eval/achievements"][..., 1]), np.zeros((2, 3, 4), dtype=bool))

    def test_jitted_eval_matches_eager_eval(self):
        env = _DummyAutoResetEvalEnv()
        config = self._make_config()
        train_state = self._make_train_state()
        evaluation_alphas = jnp.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float32,
        )
        achievement_names = ("Achievements/a", "Achievements/b")

        rng = jax.random.key(123)
        eager_rng, eager_metrics = evaluate_policy_on_alphas(
            rng=rng,
            train_state=train_state,
            env=env,
            env_params=None,
            evaluation_alphas=evaluation_alphas,
            num_eval_envs=2,
            num_eval_episodes=3,
            achievement_names=achievement_names,
            config=config,
        )
        jit_rng, jit_metrics = jax.jit(
            Partial(
                evaluate_policy_on_alphas,
                env=env,
                env_params=None,
                evaluation_alphas=evaluation_alphas,
                num_eval_envs=2,
                num_eval_episodes=3,
                achievement_names=achievement_names,
                config=config,
            )
        )(
            rng,
            train_state,
        )

        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(jit_rng)),
            np.asarray(jax.random.key_data(eager_rng)),
        )
        np.testing.assert_allclose(np.asarray(jit_metrics["eval/returns"]), np.asarray(eager_metrics["eval/returns"]), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(jit_metrics["eval/lengths"]), np.asarray(eager_metrics["eval/lengths"]))
        np.testing.assert_array_equal(np.asarray(jit_metrics["eval/achievements"]), np.asarray(eager_metrics["eval/achievements"]))


if __name__ == "__main__":
    unittest.main()
