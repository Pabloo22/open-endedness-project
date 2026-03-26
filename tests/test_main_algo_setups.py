import unittest
from types import SimpleNamespace
from unittest import mock

from crew.main_algo.config import TrainConfig
from crew.main_algo.setups import set_up_for_training


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
        "enable_wandb": False,
    }


class TestSetUpForTrainingResetWrapperComposition(unittest.TestCase):
    def _make_base_env(self):
        default_params = mock.Mock()
        default_params.replace.return_value = "env_params"
        return SimpleNamespace(default_params=default_params)

    def _run_setup(self, *, procedural_generation: bool, fixed_reset_seed: int, optimistic_reset_ratio_limit: int = 16):
        config = TrainConfig(
            **_base_config_kwargs(),
            procedural_generation=procedural_generation,
            fixed_reset_seed=fixed_reset_seed,
            optimistic_reset_ratio_limit=optimistic_reset_ratio_limit,
        )
        base_env = self._make_base_env()

        with (
            mock.patch("crew.main_algo.setups.make_craftax_env_from_name", return_value=base_env),
            mock.patch("crew.main_algo.setups.SparseCraftaxWrapper", return_value="sparse_env") as sparse_wrapper_mock,
            mock.patch("crew.main_algo.setups.FixedResetKeyEnvWrapper", return_value="fixed_env") as fixed_wrapper_mock,
            mock.patch("crew.main_algo.setups.OptimisticResetVecEnvWrapper", return_value="vec_env") as vec_wrapper_mock,
            mock.patch("crew.main_algo.setups.setup_actor_critic_train_state", return_value=("rng_after_agent", "agent")),
            mock.patch("crew.main_algo.setups.get_selected_intrinsic_modules", return_value=()),
            mock.patch("crew.main_algo.setups.setup_intrinsic_module_states", return_value=("rng_after_intrinsic", ())),
            mock.patch("crew.main_algo.setups.init_reward_normalization_stats", return_value="reward_stats"),
            mock.patch("crew.main_algo.setups.initialize_curriculum_state", return_value=("rng_final", "curriculum_state")),
        ):
            output = set_up_for_training(config)

        return output, sparse_wrapper_mock, fixed_wrapper_mock, vec_wrapper_mock

    def test_setup_wraps_env_with_fixed_reset_key_when_procedural_generation_is_disabled(self):
        output, _sparse_wrapper_mock, fixed_wrapper_mock, vec_wrapper_mock = self._run_setup(
            procedural_generation=False,
            fixed_reset_seed=777,
            optimistic_reset_ratio_limit=4,
        )

        fixed_wrapper_mock.assert_called_once_with("sparse_env", fixed_reset_seed=777)
        vec_wrapper_mock.assert_called_once()
        self.assertEqual(vec_wrapper_mock.call_args.args[0], "fixed_env")
        self.assertEqual(vec_wrapper_mock.call_args.kwargs["reset_ratio"], 16)
        self.assertEqual(output[1], "vec_env")

    def test_setup_skips_fixed_reset_wrapper_when_procedural_generation_is_enabled(self):
        output, _sparse_wrapper_mock, fixed_wrapper_mock, vec_wrapper_mock = self._run_setup(
            procedural_generation=True,
            fixed_reset_seed=777,
            optimistic_reset_ratio_limit=8,
        )

        fixed_wrapper_mock.assert_not_called()
        vec_wrapper_mock.assert_called_once()
        self.assertEqual(vec_wrapper_mock.call_args.args[0], "sparse_env")
        self.assertEqual(vec_wrapper_mock.call_args.kwargs["reset_ratio"], 8)
        self.assertEqual(output[1], "vec_env")


if __name__ == "__main__":
    unittest.main()
