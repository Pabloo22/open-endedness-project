import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from crew.main_algo.curriculum.replay_buffer import add_alpha_score_batch, init_alpha_score_replay_buffer


def _make_config(num_envs_per_batch: int, num_reward_functions: int, replay_buffer_num_batches: int):
    return SimpleNamespace(
        num_envs_per_batch=num_envs_per_batch,
        num_reward_functions=num_reward_functions,
        curriculum=SimpleNamespace(replay_buffer_num_batches=replay_buffer_num_batches),
    )


class TestAlphaScoreReplayBuffer(unittest.TestCase):
    def test_init_buffer_shapes(self):
        config = _make_config(
            num_envs_per_batch=3,
            num_reward_functions=2,
            replay_buffer_num_batches=2,
        )
        buffer = init_alpha_score_replay_buffer(config=config)

        self.assertEqual(buffer.capacity, 6)
        self.assertEqual(buffer.batch_size, 3)
        self.assertEqual(buffer.alpha.shape, (6, 2))
        self.assertEqual(buffer.score.shape, (6,))
        self.assertEqual(buffer.is_valid.shape, (6,))
        self.assertEqual(int(buffer.size), 0)
        self.assertEqual(int(buffer.write_index), 0)

    def test_insert_and_wrap_around(self):
        config = _make_config(
            num_envs_per_batch=3,
            num_reward_functions=2,
            replay_buffer_num_batches=2,
        )
        buffer = init_alpha_score_replay_buffer(config=config)

        alpha_1 = jnp.asarray([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]], dtype=jnp.float32)
        score_1 = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
        valid_1 = jnp.asarray([True, False, True], dtype=jnp.bool_)

        alpha_2 = jnp.asarray([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]], dtype=jnp.float32)
        score_2 = jnp.asarray([4.0, 5.0, 6.0], dtype=jnp.float32)
        valid_2 = jnp.asarray([True, True, False], dtype=jnp.bool_)

        alpha_3 = jnp.asarray([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7]], dtype=jnp.float32)
        score_3 = jnp.asarray([7.0, 8.0, 9.0], dtype=jnp.float32)
        valid_3 = jnp.asarray([False, True, True], dtype=jnp.bool_)

        buffer = add_alpha_score_batch(buffer, alpha_1, score_1, valid_1)
        self.assertEqual(int(buffer.size), 3)
        self.assertEqual(int(buffer.write_index), 3)
        np.testing.assert_allclose(np.asarray(buffer.alpha[:3]), np.asarray(alpha_1), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(buffer.score[:3]), np.asarray(score_1), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(buffer.is_valid[:3]), np.asarray(valid_1))

        buffer = add_alpha_score_batch(buffer, alpha_2, score_2, valid_2)
        self.assertEqual(int(buffer.size), 6)
        self.assertEqual(int(buffer.write_index), 0)
        np.testing.assert_allclose(np.asarray(buffer.alpha[3:6]), np.asarray(alpha_2), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(buffer.score[3:6]), np.asarray(score_2), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(buffer.is_valid[3:6]), np.asarray(valid_2))

        buffer = add_alpha_score_batch(buffer, alpha_3, score_3, valid_3)
        self.assertEqual(int(buffer.size), 6)
        self.assertEqual(int(buffer.write_index), 3)
        np.testing.assert_allclose(np.asarray(buffer.alpha[:3]), np.asarray(alpha_3), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(buffer.score[:3]), np.asarray(score_3), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(buffer.is_valid[:3]), np.asarray(valid_3))
        np.testing.assert_allclose(np.asarray(buffer.alpha[3:6]), np.asarray(alpha_2), rtol=0, atol=1e-6)

    def test_jit_insert_matches_eager(self):
        config = _make_config(
            num_envs_per_batch=2,
            num_reward_functions=3,
            replay_buffer_num_batches=2,
        )
        buffer = init_alpha_score_replay_buffer(config=config)
        alpha = jnp.asarray([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], dtype=jnp.float32)
        score = jnp.asarray([0.9, 1.1], dtype=jnp.float32)
        valid = jnp.asarray([True, False], dtype=jnp.bool_)

        eager_buffer = add_alpha_score_batch(buffer, alpha, score, valid)
        jit_buffer = jax.jit(add_alpha_score_batch)(buffer, alpha, score, valid)

        np.testing.assert_allclose(np.asarray(jit_buffer.alpha), np.asarray(eager_buffer.alpha), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(jit_buffer.score), np.asarray(eager_buffer.score), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(jit_buffer.is_valid), np.asarray(eager_buffer.is_valid))
        self.assertEqual(int(jit_buffer.size), int(eager_buffer.size))
        self.assertEqual(int(jit_buffer.write_index), int(eager_buffer.write_index))


if __name__ == "__main__":
    unittest.main()
