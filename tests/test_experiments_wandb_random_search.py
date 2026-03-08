import unittest

from crew.experiments.wandb_random_search import (
    DEFAULT_OBJECTIVE_METRIC,
    build_base_tuning_config,
    build_default_sweep_config,
    build_trial_config_from_overrides,
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

    def test_build_default_sweep_config_uses_random_search(self):
        sweep_config = build_default_sweep_config(training_mode="curriculum")

        self.assertEqual(sweep_config["method"], "random")
        self.assertEqual(sweep_config["metric"]["name"], DEFAULT_OBJECTIVE_METRIC)
        self.assertEqual(sweep_config["metric"]["goal"], "maximize")
        self.assertIn("curriculum.score_lambda", sweep_config["parameters"])
        self.assertIn("rnd.predictor_network_lr", sweep_config["parameters"])


if __name__ == "__main__":
    unittest.main()
