import unittest

import jax
import jax.numpy as jnp
import numpy as np

from curemix.main_algo.curriculum.score_estimation import compute_scores


class TestScoreEstimation(unittest.TestCase):
    def test_family2_score_alp_mode(self):
        alpha_batch = jnp.asarray(
            [
                [0.7, 0.3],
                [0.1, 0.9],
            ],
            dtype=jnp.float32,
        )
        lp_per_reward_function = jnp.asarray(
            [
                [2.0, -4.0],
                [-3.0, 5.0],
            ],
            dtype=jnp.float32,
        )

        scores, metrics = compute_scores(
            alpha_batch=alpha_batch,
            lp_per_reward_function=lp_per_reward_function,
            score_lp_mode="alp",
            score_lambda=0.5,
        )

        lp_used = np.abs(np.asarray(lp_per_reward_function))
        lp_total = np.sum(np.asarray(alpha_batch) * lp_used, axis=1)
        expected_scores = 0.5 * lp_total + 0.5 * lp_used[:, 0]
        np.testing.assert_allclose(np.asarray(scores), expected_scores, rtol=0, atol=1e-6)
        self.assertEqual(set(metrics.keys()), {"curriculum/score_mean"})
        np.testing.assert_allclose(
            np.asarray(metrics["curriculum/score_mean"]),
            np.asarray(expected_scores.mean()),
            rtol=0,
            atol=1e-6,
        )

    def test_family2_score_lp_mode_clips_negative_lp(self):
        alpha_batch = jnp.asarray(
            [
                [0.5, 0.5],
                [0.2, 0.8],
            ],
            dtype=jnp.float32,
        )
        lp_per_reward_function = jnp.asarray(
            [
                [1.0, -5.0],
                [-2.0, 4.0],
            ],
            dtype=jnp.float32,
        )

        scores, metrics = compute_scores(
            alpha_batch=alpha_batch,
            lp_per_reward_function=lp_per_reward_function,
            score_lp_mode="lp",
            score_lambda=0.5,
        )

        lp_used = np.clip(np.asarray(lp_per_reward_function), a_min=0.0, a_max=None)
        lp_total = np.sum(np.asarray(alpha_batch) * lp_used, axis=1)
        expected_scores = 0.5 * lp_total + 0.5 * lp_used[:, 0]
        np.testing.assert_allclose(np.asarray(scores), expected_scores, rtol=0, atol=1e-6)
        self.assertEqual(set(metrics.keys()), {"curriculum/score_mean"})

    def test_jit_matches_eager(self):
        alpha_batch = jnp.asarray(
            [
                [0.3, 0.7],
                [0.9, 0.1],
            ],
            dtype=jnp.float32,
        )
        lp_per_reward_function = jnp.asarray(
            [
                [-1.0, 3.0],
                [2.0, -2.0],
            ],
            dtype=jnp.float32,
        )
        eager_scores, eager_metrics = compute_scores(
            alpha_batch=alpha_batch,
            lp_per_reward_function=lp_per_reward_function,
            score_lp_mode="alp",
            score_lambda=0.5,
        )
        jit_scores, jit_metrics = jax.jit(
            compute_scores,
            static_argnames=("score_lp_mode",),
        )(
            alpha_batch=alpha_batch,
            lp_per_reward_function=lp_per_reward_function,
            score_lp_mode="alp",
            score_lambda=0.5,
        )

        np.testing.assert_allclose(np.asarray(jit_scores), np.asarray(eager_scores), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(jit_metrics["curriculum/score_mean"]),
            np.asarray(eager_metrics["curriculum/score_mean"]),
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
