import unittest
from types import SimpleNamespace
from unittest import mock

from crew.experiments.run_training import run_main_algo_training


class TestMainAlgoRunTrainingDispatch(unittest.TestCase):
    def _setup_tuple(self):
        return (
            "rng",
            "env",
            "env_params",
            "agent_train_state",
            "reward_normalization_stats",
            (),
            (),
            "curriculum_state",
        )

    def test_curriculum_mode_dispatches_to_curriculum_loop(self):
        config = SimpleNamespace(
            training_mode="curriculum",
            train_seed=0,
            env_id="Craftax-Classic-Symbolic-v1",
            artifacts_root="/tmp",
        )
        curriculum_out = {
            "agent_state": SimpleNamespace(params={}),
            "intrinsic_states": (),
            "curriculum_state": "curriculum_state",
            "reward_normalization_stats": "stats",
            "metrics": {},
        }

        with (
            mock.patch("crew.experiments.run_training.set_up_for_training", return_value=self._setup_tuple()),
            mock.patch("crew.experiments.run_training.full_training", return_value=curriculum_out) as curriculum_mock,
            mock.patch("crew.experiments.run_training.full_training_baseline") as baseline_mock,
            mock.patch("crew.experiments.run_training.jax.block_until_ready", side_effect=lambda x: x),
        ):
            out = run_main_algo_training(config=config, save_results=False)

        self.assertIs(out, curriculum_out)
        self.assertEqual(curriculum_mock.call_count, 1)
        self.assertEqual(baseline_mock.call_count, 0)

    def test_baseline_mode_dispatches_to_baseline_loop(self):
        config = SimpleNamespace(
            training_mode="baseline",
            train_seed=0,
            env_id="Craftax-Classic-Symbolic-v1",
            artifacts_root="/tmp",
        )
        baseline_out = {
            "agent_state": SimpleNamespace(params={}),
            "intrinsic_states": (),
            "reward_normalization_stats": "stats",
            "metrics": {},
        }

        with (
            mock.patch("crew.experiments.run_training.set_up_for_training", return_value=self._setup_tuple()),
            mock.patch("crew.experiments.run_training.full_training") as curriculum_mock,
            mock.patch("crew.experiments.run_training.full_training_baseline", return_value=baseline_out) as baseline_mock,
            mock.patch("crew.experiments.run_training.jax.block_until_ready", side_effect=lambda x: x),
        ):
            out = run_main_algo_training(config=config, save_results=False)

        self.assertIs(out, baseline_out)
        self.assertEqual(curriculum_mock.call_count, 0)
        self.assertEqual(baseline_mock.call_count, 1)

    def test_save_path_is_built_from_config(self):
        curriculum_config = SimpleNamespace(
            training_mode="curriculum",
            train_seed=0,
            env_id="Craftax-Classic-Symbolic-v1",
            artifacts_root="/tmp",
            selected_intrinsic_modules=("rnd",),
        )
        baseline_config = SimpleNamespace(
            training_mode="baseline",
            train_seed=0,
            env_id="Craftax-Classic-Symbolic-v1",
            artifacts_root="/tmp",
            selected_intrinsic_modules=(),
        )
        curriculum_out = {
            "agent_state": SimpleNamespace(params={}),
            "intrinsic_states": (),
            "curriculum_state": "curriculum_state",
            "reward_normalization_stats": "stats",
            "metrics": {},
        }
        baseline_out = {
            "agent_state": SimpleNamespace(params={}),
            "intrinsic_states": (),
            "reward_normalization_stats": "stats",
            "metrics": {},
        }

        with (
            mock.patch("crew.experiments.run_training.set_up_for_training", return_value=self._setup_tuple()),
            mock.patch("crew.experiments.run_training.full_training", return_value=curriculum_out),
            mock.patch("crew.experiments.run_training.full_training_baseline", return_value=baseline_out),
            mock.patch("crew.experiments.run_training.jax.block_until_ready", side_effect=lambda x: x),
            mock.patch("crew.experiments.run_training.build_trained_weights_path") as build_path_mock,
            mock.patch("crew.experiments.run_training.orbax.PyTreeCheckpointer"),
            mock.patch("crew.experiments.run_training.orbax_utils.save_args_from_target", return_value=None),
            mock.patch("crew.experiments.run_training.flax.core.freeze", side_effect=lambda x: x),
        ):
            path_obj = mock.MagicMock()
            path_obj.parent.mkdir = mock.MagicMock()
            build_path_mock.return_value = path_obj

            run_main_algo_training(config=curriculum_config, save_results=True)
            run_main_algo_training(config=baseline_config, save_results=True)

        called_configs = [call.kwargs["config"] for call in build_path_mock.call_args_list]
        self.assertEqual(called_configs, [curriculum_config, baseline_config])


if __name__ == "__main__":
    unittest.main()
