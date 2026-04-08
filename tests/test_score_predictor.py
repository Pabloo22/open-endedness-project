import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

from curemix.main_algo.curriculum.replay_buffer import add_alpha_score_batch, init_alpha_score_replay_buffer
from curemix.main_algo.curriculum.score_predictor import (
    init_score_predictor_train_state,
    train_score_predictor_on_buffer,
)


def _make_config(
    num_envs_per_batch: int = 4,
    num_reward_functions: int = 2,
    replay_buffer_num_batches: int = 2,
):
    return SimpleNamespace(
        num_envs_per_batch=num_envs_per_batch,
        num_reward_functions=num_reward_functions,
        adam_eps=1e-5,
        curriculum=SimpleNamespace(
            replay_buffer_num_batches=replay_buffer_num_batches,
            predictor_lr=1e-3,
            predictor_update_epochs=2,
            predictor_num_minibatches=4,
            predictor_hidden_dim=32,
            predictor_activation="relu",
            importance_num_candidates_multiplier=8,
            min_batches_for_predictor_sampling=1,
            sampling_weights_eps=1e-8,
        ),
    )


def _tree_allclose(tree_a, tree_b, atol: float = 1e-6) -> bool:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y, atol=atol), tree_a, tree_b)
    )
    return all(bool(v) for v in leaves)


class TestScorePredictor(unittest.TestCase):
    def test_predictor_init_output_shape(self):
        config = _make_config()
        train_state = init_score_predictor_train_state(
            rng=jax.random.key(0),
            config=config,
        )
        alpha = jnp.asarray(
            [
                [0.2, 0.8],
                [0.7, 0.3],
            ],
            dtype=jnp.float32,
        )
        predictions = train_state.apply_fn(train_state.params, alpha)
        self.assertEqual(predictions.shape, (2,))

    def test_all_invalid_mask_keeps_params_unchanged(self):
        config = _make_config()
        replay_buffer = init_alpha_score_replay_buffer(config=config)
        alpha_batch = jnp.asarray(
            [
                [0.1, 0.9],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.9, 0.1],
            ],
            dtype=jnp.float32,
        )
        score_batch = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        invalid_mask = jnp.zeros((config.num_envs_per_batch,), dtype=jnp.bool_)
        replay_buffer = add_alpha_score_batch(replay_buffer, alpha_batch, score_batch, invalid_mask)

        train_state = init_score_predictor_train_state(
            rng=jax.random.key(1),
            config=config,
        )
        original_params = train_state.params
        rng, updated_state, metrics = train_score_predictor_on_buffer(
            rng=jax.random.key(2),
            score_predictor_train_state=train_state,
            alpha_score_replay_buffer=replay_buffer,
            config=config,
        )
        del rng
        self.assertTrue(_tree_allclose(original_params, updated_state.params))
        self.assertAlmostEqual(float(metrics["curriculum/predictor_loss"]), 0.0, places=7)

    def test_training_updates_params_with_valid_targets_and_jit(self):
        config = _make_config()
        replay_buffer = init_alpha_score_replay_buffer(config=config)

        alpha_batch_1 = jnp.asarray(
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ],
            dtype=jnp.float32,
        )
        alpha_batch_2 = jnp.asarray(
            [
                [0.5, 0.5],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.8, 0.2],
            ],
            dtype=jnp.float32,
        )
        score_batch_1 = 2.0 * alpha_batch_1[:, 0] + 0.5 * alpha_batch_1[:, 1]
        score_batch_2 = 2.0 * alpha_batch_2[:, 0] + 0.5 * alpha_batch_2[:, 1]
        valid_mask = jnp.ones((config.num_envs_per_batch,), dtype=jnp.bool_)

        replay_buffer = add_alpha_score_batch(replay_buffer, alpha_batch_1, score_batch_1, valid_mask)
        replay_buffer = add_alpha_score_batch(replay_buffer, alpha_batch_2, score_batch_2, valid_mask)

        train_state = init_score_predictor_train_state(
            rng=jax.random.key(3),
            config=config,
        )
        original_params = train_state.params

        eager_rng, eager_state, eager_metrics = train_score_predictor_on_buffer(
            rng=jax.random.key(4),
            score_predictor_train_state=train_state,
            alpha_score_replay_buffer=replay_buffer,
            config=config,
        )
        self.assertFalse(_tree_allclose(original_params, eager_state.params))
        self.assertTrue(np.isfinite(float(eager_metrics["curriculum/predictor_loss"])))

        jit_rng, jit_state, jit_metrics = jax.jit(
            Partial(train_score_predictor_on_buffer, config=config),
        )(
            jax.random.key(4),
            train_state,
            replay_buffer,
        )

        np.testing.assert_array_equal(
            np.asarray(jax.random.key_data(jit_rng)),
            np.asarray(jax.random.key_data(eager_rng)),
        )
        self.assertTrue(_tree_allclose(jit_state.params, eager_state.params))
        np.testing.assert_allclose(
            np.asarray(jit_metrics["curriculum/predictor_loss"]),
            np.asarray(eager_metrics["curriculum/predictor_loss"]),
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
