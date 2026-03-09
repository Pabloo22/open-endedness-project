import unittest

import numpy as np

from crew.experiments.wandb_random_search import (
    DEFAULT_OBJECTIVE_METRIC,
    _derive_trial_seed,
    build_base_tuning_config,
    build_default_sweep_config,
    build_trial_config_from_overrides,
    extract_trial_summary,
)


class TestWandbRandomSearch(unittest.TestCase):
    def test_build_base_tuning_config_curriculum_sets_expected_defaults(self):
        config = build_base_tuning_config(
            training_mode="curriculum",
            project="proj",
            entity="entity",
            group="group",
            train_seed=7,
            total_timesteps=1234,
        )

        self.assertEqual(config.training_mode, "curriculum")
        self.assertEqual(config.selected_intrinsic_modules, ("rnd",))
        self.assertEqual(config.evaluation_alphas, ((0.8, 0.2), (1.0, 0.0)))
        self.assertFalse(config.enable_wandb)
        self.assertEqual(config.wandb_project, "proj")
        self.assertEqual(config.train_seed, 7)
        self.assertEqual(config.total_timesteps, 1234)

    def test_build_trial_config_from_overrides_updates_nested_fields_and_qkv(self):
        base_config = build_base_tuning_config(
            training_mode="curriculum",
            project="proj",
            entity=None,
            group=None,
            train_seed=0,
            total_timesteps=2048,
        )

        trial_config = build_trial_config_from_overrides(
            base_config,
            {
                "lr": 5e-4,
                "transformer_hidden_states_dim": 128,
                "curriculum.score_lambda": 0.75,
                "rnd.predictor_network_lr": 2e-4,
            },
        )

        self.assertEqual(trial_config.lr, 5e-4)
        self.assertEqual(trial_config.transformer_hidden_states_dim, 128)
        self.assertEqual(trial_config.qkv_features, 128)
        self.assertEqual(trial_config.curriculum.score_lambda, 0.75)
        self.assertEqual(trial_config.rnd.predictor_network_lr, 2e-4)

    def test_build_trial_config_from_overrides_updates_train_seed_when_provided(self):
        base_config = build_base_tuning_config(
            training_mode="baseline",
            project="proj",
            entity=None,
            group=None,
            train_seed=7,
            total_timesteps=2048,
        )

        trial_config = build_trial_config_from_overrides(base_config, {"train_seed": 123})

        self.assertEqual(trial_config.train_seed, 123)

    def test_derive_trial_seed_is_deterministic_and_unique_per_run(self):
        first_seed = _derive_trial_seed(base_seed=11, run_id="abc123")
        second_seed = _derive_trial_seed(base_seed=11, run_id="def456")

        self.assertEqual(first_seed, _derive_trial_seed(base_seed=11, run_id="abc123"))
        self.assertNotEqual(first_seed, second_seed)

    def test_build_default_sweep_config_uses_random_search(self):
        sweep_config = build_default_sweep_config(training_mode="curriculum")

        self.assertEqual(sweep_config["method"], "random")
        self.assertEqual(sweep_config["metric"]["name"], DEFAULT_OBJECTIVE_METRIC)
        self.assertEqual(sweep_config["metric"]["goal"], "maximize")
        self.assertIn("curriculum.score_lambda", sweep_config["parameters"])
        self.assertIn("rnd.predictor_network_lr", sweep_config["parameters"])

    def test_extract_trial_summary_uses_final_metric_history_entries(self):
        train_info = {
            "metrics": {
                "eval/returns": np.asarray(
                    [
                        [[[1.0, 2.0]], [[3.0, 4.0]]],
                        [[[10.0, 20.0]], [[30.0, 40.0]]],
                    ],
                    dtype=np.float32,
                ),
                "eval/lengths": np.asarray(
                    [
                        [[[5.0, 6.0]], [[7.0, 8.0]]],
                        [[[50.0, 60.0]], [[70.0, 80.0]]],
                    ],
                    dtype=np.float32,
                ),
                "eval/achievements": np.asarray(
                    [
                        [[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 1.0], [0.0, 0.0]]]],
                        [[[[1.0, 1.0], [1.0, 0.0]]], [[[0.0, 1.0], [1.0, 1.0]]]],
                    ],
                    dtype=np.float32,
                ),
                "run/total_env_steps": np.asarray([100, 200], dtype=np.int32),
                "ppo/total_loss": np.asarray([1.5, 2.5], dtype=np.float32),
                "ppo/actor_loss": np.asarray([0.5, 0.75], dtype=np.float32),
                "ppo/entropy": np.asarray([0.2, 0.1], dtype=np.float32),
                "ppo/approx_kl": np.asarray([0.01, 0.02], dtype=np.float32),
                "curriculum/score_mean": np.asarray([0.3, 0.8], dtype=np.float32),
                "intrinsic_modules/rnd/predictor_loss": np.asarray([4.0, 3.0], dtype=np.float32),
            }
        }

        summary = extract_trial_summary(train_info)

        self.assertEqual(summary["tuning/final_total_env_steps"], 200)
        self.assertAlmostEqual(summary["tuning/final_total_loss"], 2.5)
        self.assertAlmostEqual(summary["tuning/final_actor_loss"], 0.75)
        self.assertAlmostEqual(summary["tuning/final_entropy"], 0.1)
        self.assertAlmostEqual(summary["tuning/final_approx_kl"], 0.02)
        self.assertAlmostEqual(summary["tuning/final_curriculum_score_mean"], 0.8)
        self.assertAlmostEqual(summary["tuning/final_rnd_predictor_loss"], 3.0)
        self.assertAlmostEqual(summary[DEFAULT_OBJECTIVE_METRIC], 25.0)
        self.assertAlmostEqual(summary["tuning/objective_eval_return_alpha0_mean"], 15.0)
        self.assertAlmostEqual(summary["tuning/final_eval_length_mean"], 65.0)
        self.assertAlmostEqual(summary["tuning/final_eval_num_achievements_mean"], 1.5)


if __name__ == "__main__":
    unittest.main()
