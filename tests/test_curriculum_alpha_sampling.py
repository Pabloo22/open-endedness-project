import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

from curemix.main_algo.curriculum.alpha_sampling import sample_alpha_batch
from curemix.main_algo.curriculum.lp_normalization import init_lp_normalization_stats
from curemix.main_algo.curriculum.replay_buffer import init_alpha_score_replay_buffer
from curemix.main_algo.curriculum.score_predictor import init_score_predictor_train_state
from curemix.main_algo.types import CurriculumState


def _make_config(
    num_envs_per_batch: int = 4,
    num_reward_functions: int = 2,
):
    return SimpleNamespace(
        num_envs_per_batch=num_envs_per_batch,
        num_reward_functions=num_reward_functions,
        adam_eps=1e-5,
        curriculum=SimpleNamespace(
            replay_buffer_num_batches=3,
            predictor_lr=1e-3,
            predictor_update_epochs=1,
            predictor_num_minibatches=3,
            predictor_hidden_dim=32,
            predictor_activation="relu",
            importance_num_candidates_multiplier=5,
            min_batches_for_predictor_sampling=1,
            sampling_weights_eps=1e-8,
        ),
    )


class TestCurriculumAlphaSampling(unittest.TestCase):
    def test_warmup_path_samples_uniform_and_preserves_simplex(self):
        config = _make_config()
        replay_buffer = init_alpha_score_replay_buffer(config=config)
        predictor_train_state = init_score_predictor_train_state(
            rng=jax.random.key(0),
            config=config,
        )
        lp_normalization_stats = init_lp_normalization_stats(config.num_reward_functions)
        curriculum_state = CurriculumState(
            alpha_score_replay_buffer=replay_buffer,
            score_predictor_train_state=predictor_train_state,
            lp_normalization_stats=lp_normalization_stats,
            num_batches_seen=jnp.array(0, dtype=jnp.int32),
        )

        _, alpha_batch, metrics = sample_alpha_batch(
            rng=jax.random.key(1),
            curriculum_state=curriculum_state,
            config=config,
        )

        self.assertEqual(alpha_batch.shape, (config.num_envs_per_batch, config.num_reward_functions))
        np.testing.assert_allclose(np.asarray(alpha_batch.sum(axis=1)), np.ones((config.num_envs_per_batch,)), rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.asarray(alpha_batch) >= 0.0))
        self.assertEqual(set(metrics.keys()), {"curriculum/pred_score_mean"})
        self.assertAlmostEqual(float(metrics["curriculum/pred_score_mean"]), 0.0, places=7)

    def test_predictor_path_runs_after_min_batch_threshold(self):
        config = _make_config()
        replay_buffer = init_alpha_score_replay_buffer(config=config)
        predictor_train_state = init_score_predictor_train_state(
            rng=jax.random.key(2),
            config=config,
        )
        lp_normalization_stats = init_lp_normalization_stats(config.num_reward_functions)
        curriculum_state = CurriculumState(
            alpha_score_replay_buffer=replay_buffer,
            score_predictor_train_state=predictor_train_state,
            lp_normalization_stats=lp_normalization_stats,
            num_batches_seen=jnp.array(1, dtype=jnp.int32),
        )

        _, alpha_batch, metrics = sample_alpha_batch(
            rng=jax.random.key(3),
            curriculum_state=curriculum_state,
            config=config,
        )

        self.assertEqual(alpha_batch.shape, (config.num_envs_per_batch, config.num_reward_functions))
        np.testing.assert_allclose(np.asarray(alpha_batch.sum(axis=1)), np.ones((config.num_envs_per_batch,)), rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.asarray(alpha_batch) >= 0.0))
        self.assertEqual(set(metrics.keys()), {"curriculum/pred_score_mean"})
        self.assertTrue(np.isfinite(float(metrics["curriculum/pred_score_mean"])))

    def test_jitted_sampler_handles_both_branches(self):
        config = _make_config()
        replay_buffer = init_alpha_score_replay_buffer(config=config)
        predictor_train_state = init_score_predictor_train_state(
            rng=jax.random.key(4),
            config=config,
        )
        lp_normalization_stats = init_lp_normalization_stats(config.num_reward_functions)
        warmup_state = CurriculumState(
            alpha_score_replay_buffer=replay_buffer,
            score_predictor_train_state=predictor_train_state,
            lp_normalization_stats=lp_normalization_stats,
            num_batches_seen=jnp.array(0, dtype=jnp.int32),
        )
        predictor_state = warmup_state.replace(num_batches_seen=jnp.array(1, dtype=jnp.int32))

        sampler_jit = jax.jit(
            Partial(sample_alpha_batch, config=config),
        )
        _, _, warmup_metrics = sampler_jit(jax.random.key(5), warmup_state)
        _, _, predictor_metrics = sampler_jit(jax.random.key(6), predictor_state)

        self.assertAlmostEqual(float(warmup_metrics["curriculum/pred_score_mean"]), 0.0, places=7)
        self.assertTrue(np.isfinite(float(predictor_metrics["curriculum/pred_score_mean"])))


if __name__ == "__main__":
    unittest.main()
