import unittest
from unittest.mock import patch

import numpy as np

from crew.experiments import tuning_configs
from crew.experiments.wandb_random_search import (
    DEFAULT_OBJECTIVE_METRIC,
    DEFAULT_BASELINE_INTRINSIC_ALPHA,
    SWEEP_METHOD_GRID,
    SWEEP_METHOD_RANDOM,
    TUNING_PHASE_CURRICULUM,
    TUNING_PHASE_GENERIC,
    TUNING_PHASE_INTRINSIC,
    _derive_trial_seed,
    build_base_tuning_config,
    build_default_sweep_config,
    build_trial_config_from_overrides,
    extract_trial_summary,
    parse_fixed_overrides,
)


class TestWandbRandomSearch(unittest.TestCase):
    def test_build_base_tuning_config_curriculum_sets_expected_defaults(self):
        config = build_base_tuning_config(
            tuning_phase=TUNING_PHASE_CURRICULUM,
            project="proj",
            entity="entity",
            group="group",
            train_seed=7,
            total_timesteps=1234,
            intrinsic_modules=("rnd",),
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
            tuning_phase=TUNING_PHASE_CURRICULUM,
            project="proj",
            entity=None,
            group=None,
            train_seed=0,
            total_timesteps=2048,
            intrinsic_modules=("rnd",),
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
            tuning_phase=TUNING_PHASE_GENERIC,
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

    def test_build_default_sweep_config_curriculum_phase_uses_only_curriculum_params(self):
        sweep_config = build_default_sweep_config(tuning_phase=TUNING_PHASE_CURRICULUM, intrinsic_modules=("rnd",))

        self.assertEqual(sweep_config["method"], SWEEP_METHOD_RANDOM)
        self.assertEqual(sweep_config["metric"]["name"], DEFAULT_OBJECTIVE_METRIC)
        self.assertEqual(sweep_config["metric"]["goal"], "maximize")
        self.assertIn("curriculum.score_lambda", sweep_config["parameters"])
        self.assertNotIn("rnd.predictor_network_lr", sweep_config["parameters"])
        self.assertNotIn("lr", sweep_config["parameters"])

    def test_build_default_sweep_config_grid_method_is_reflected_in_config(self):
        sweep_config = build_default_sweep_config(
            tuning_phase=TUNING_PHASE_CURRICULUM,
            intrinsic_modules=("rnd",),
            method=SWEEP_METHOD_GRID,
        )

        self.assertEqual(sweep_config["method"], SWEEP_METHOD_GRID)

    def test_build_default_sweep_config_raises_on_unsupported_method(self):
        with self.assertRaises(ValueError):
            build_default_sweep_config(tuning_phase=TUNING_PHASE_GENERIC, method="bayes")

    def test_build_default_sweep_config_generic_phase_uses_only_shared_params(self):
        sweep_config = build_default_sweep_config(tuning_phase=TUNING_PHASE_GENERIC)

        self.assertIn("lr", sweep_config["parameters"])
        self.assertNotIn("curriculum.score_lambda", sweep_config["parameters"])
        self.assertNotIn("rnd.predictor_network_lr", sweep_config["parameters"])

    def test_build_default_sweep_config_intrinsic_phase_uses_only_intrinsic_params(self):
        sweep_config = build_default_sweep_config(tuning_phase=TUNING_PHASE_INTRINSIC, intrinsic_modules=("rnd",))

        self.assertIn("rnd.predictor_network_lr", sweep_config["parameters"])
        self.assertIn("baseline_fixed_training_alpha", sweep_config["parameters"])
        self.assertNotIn("curriculum.score_lambda", sweep_config["parameters"])
        self.assertNotIn("lr", sweep_config["parameters"])

    def test_build_base_tuning_config_intrinsic_phase_sets_baseline_with_intrinsic_module(self):
        config = build_base_tuning_config(
            tuning_phase=TUNING_PHASE_INTRINSIC,
            intrinsic_modules=("rnd",),
            project="proj",
            entity=None,
            group=None,
            train_seed=5,
            total_timesteps=2048,
        )

        self.assertEqual(config.training_mode, "baseline")
        self.assertEqual(config.selected_intrinsic_modules, ("rnd",))
        self.assertEqual(
            config.baseline_fixed_training_alpha,
            (1.0 - DEFAULT_BASELINE_INTRINSIC_ALPHA, DEFAULT_BASELINE_INTRINSIC_ALPHA),
        )

    def test_build_base_tuning_config_intrinsic_phase_rejects_multiple_modules(self):
        with self.assertRaisesRegex(ValueError, "expects exactly one intrinsic module"):
            build_base_tuning_config(
                tuning_phase=TUNING_PHASE_INTRINSIC,
                intrinsic_modules=("rnd", "icm"),
                project="proj",
                entity=None,
                group=None,
                train_seed=5,
                total_timesteps=2048,
            )

    def test_curriculum_preset_lookup_accepts_multiple_modules_when_preset_exists(self):
        curriculum_config = {
            **tuning_configs.RND_CURRICULUM_BASE_CONFIG_V1,
            "selected_intrinsic_modules": ("rnd", "icm"),
            "evaluation_alphas": ((0.7, 0.2, 0.1), (1.0, 0.0, 0.0)),
        }

        with patch.dict(
            tuning_configs.ACTIVE_CURRICULUM_BASE_CONFIGS,
            {("rnd", "icm"): curriculum_config},
            clear=False,
        ):
            resolved_config = tuning_configs.get_curriculum_base_config_for_modules(("rnd", "icm"))

        self.assertEqual(resolved_config["training_mode"], "curriculum")
        self.assertEqual(resolved_config["selected_intrinsic_modules"], ("rnd", "icm"))
        self.assertEqual(resolved_config["evaluation_alphas"], ((0.7, 0.2, 0.1), (1.0, 0.0, 0.0)))

    def test_parse_fixed_overrides_supports_scalars_and_collections(self):
        parsed = parse_fixed_overrides(
            [
                "num_envs_per_batch=128",
                "baseline_fixed_training_alpha=[0.8, 0.2]",
                "wandb_group='phase-1'",
            ]
        )

        self.assertEqual(parsed["num_envs_per_batch"], 128)
        self.assertEqual(parsed["baseline_fixed_training_alpha"], [0.8, 0.2])
        self.assertEqual(parsed["wandb_group"], "phase-1")

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
