import unittest

import numpy as np

from crew.main_algo.config import TrainConfig


def _base_config_kwargs() -> dict:
    return {
        "total_timesteps": 64,
        "num_envs_per_batch": 16,
        "num_steps_per_env": 8,
        "num_steps_per_update": 8,
        "num_minibatches": 4,
        "past_context_length": 4,
        "subsequence_length_in_loss_calculation": 4,
        "num_transformer_blocks": 1,
        "transformer_hidden_states_dim": 16,
        "qkv_features": 16,
        "head_hidden_dim": 16,
    }


class TestMainAlgoEvalConfig(unittest.TestCase):
    def test_evaluation_alphas_are_validated_and_materialized(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            eval_every_n_batches=2,
            eval_num_episodes=7,
            evaluation_alphas=((1.0, 0.0), (0.3, 0.7)),
        )

        self.assertEqual(config.evaluation_alphas_array.shape, (2, config.num_reward_functions))
        np.testing.assert_allclose(
            np.asarray(config.evaluation_alphas_array.sum(axis=1)), np.ones((2,)), rtol=0, atol=1e-6
        )
        self.assertEqual(config.eval_num_episodes, 7)

    def test_default_evaluation_alphas_is_extrinsic_only(self):
        config = TrainConfig(
            **_base_config_kwargs(),
        )

        self.assertEqual(config.evaluation_alphas_array.shape, (1, config.num_reward_functions))
        np.testing.assert_allclose(
            np.asarray(config.evaluation_alphas_array[0]),
            np.asarray([1.0] + [0.0] * (config.num_reward_functions - 1)),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(config.evaluation_alpha_labels, ("ext10_rnd00",))

    def test_eval_num_episodes_is_set(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            eval_num_episodes=9,
        )
        self.assertEqual(config.eval_num_episodes, 9)

    def test_is_timing_run_defaults_to_false_and_can_be_enabled(self):
        default_config = TrainConfig(
            **_base_config_kwargs(),
        )
        self.assertFalse(default_config.is_timing_run)

        timing_config = TrainConfig(
            **_base_config_kwargs(),
            is_timing_run=True,
        )
        self.assertTrue(timing_config.is_timing_run)

    def test_invalid_evaluation_alphas_raise(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                evaluation_alphas=((1.0,),),
            )

        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                evaluation_alphas=((-0.1, 1.1),),
            )

        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                evaluation_alphas=((0.2, 0.2),),
            )

        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                eval_every_n_batches=0,
            )

    def test_default_evaluation_alpha_labels_are_derived_from_alphas(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            evaluation_alphas=((0.8, 0.2), (0.0, 1.0), (1.0, 0.0)),
        )
        self.assertEqual(
            config.evaluation_alpha_labels,
            ("ext08_rnd02", "ext00_rnd10", "ext10_rnd00"),
        )

    def test_default_evaluation_alpha_labels_are_made_unique_when_deciles_collide(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            evaluation_alphas=((0.81, 0.19), (0.84, 0.16)),
        )
        self.assertEqual(
            config.evaluation_alpha_labels,
            ("ext08_rnd02", "ext08_rnd02_2"),
        )

    def test_passing_evaluation_alpha_labels_is_not_supported(self):
        with self.assertRaises(TypeError):
            TrainConfig(
                **_base_config_kwargs(),
                evaluation_alphas=((1.0, 0.0), (0.3, 0.7)),
                evaluation_alpha_labels=("ext_only", "mix"),
            )

    def test_baseline_mode_allows_empty_intrinsic_modules_and_defaults_to_extrinsic_only_alpha(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            training_mode="baseline",
            selected_intrinsic_modules=(),
        )
        self.assertEqual(config.num_reward_functions, 1)
        self.assertEqual(config.baseline_fixed_training_alpha, (1.0,))
        np.testing.assert_allclose(
            np.asarray(config.evaluation_alphas_array),
            np.asarray([[1.0]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        self.assertFalse(config.inject_alpha_at_trunk)
        self.assertFalse(config.inject_alpha_at_actor_head)
        self.assertFalse(config.inject_alpha_at_critic_head)

    def test_baseline_mode_uses_fixed_training_alpha_for_eval_and_ignores_evaluation_alphas(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            training_mode="baseline",
            selected_intrinsic_modules=("rnd",),
            baseline_fixed_training_alpha=(0.6, 0.4),
            evaluation_alphas=((1.0, 0.0),),
        )
        self.assertEqual(config.baseline_fixed_training_alpha, (0.6, 0.4))
        np.testing.assert_allclose(
            np.asarray(config.evaluation_alphas_array),
            np.asarray([[0.6, 0.4]], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(config.evaluation_alpha_labels, ("ext06_rnd04",))

    def test_invalid_baseline_fixed_training_alpha_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                training_mode="baseline",
                selected_intrinsic_modules=("rnd",),
                baseline_fixed_training_alpha=(0.7,),
            )

        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                training_mode="baseline",
                selected_intrinsic_modules=("rnd",),
                baseline_fixed_training_alpha=(0.7, 0.7),
            )

        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                training_mode="baseline",
                selected_intrinsic_modules=("rnd",),
                baseline_fixed_training_alpha=(-0.1, 1.1),
            )


if __name__ == "__main__":
    unittest.main()
