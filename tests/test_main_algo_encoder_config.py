import unittest

from crew.main_algo.config import NGUConfig, RNDConfig, TrainConfig


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


class TestMainAlgoEncoderConfig(unittest.TestCase):
    def test_encoder_mode_defaults_to_flat_symbolic(self):
        config = TrainConfig(**_base_config_kwargs())
        self.assertEqual(config.encoder_mode, "flat_symbolic")

    def test_craftax_structured_encoder_mode_is_supported(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            encoder_mode="craftax_structured",
        )
        self.assertEqual(config.encoder_mode, "craftax_structured")

    def test_invalid_encoder_mode_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                encoder_mode="not_a_real_mode",
            )

    def test_nested_intrinsic_encoder_mode_fields_are_removed(self):
        with self.assertRaises(TypeError):
            RNDConfig(encoder_mode="flat_symbolic")

        with self.assertRaises(TypeError):
            NGUConfig(encoder_mode="flat_symbolic")


if __name__ == "__main__":
    unittest.main()
