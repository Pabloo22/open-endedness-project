import unittest

from crew.main_algo.config import ICMConfig, NGUConfig, RNDConfig, TrainConfig


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
        self.assertEqual(config.rnd.encoder_mode, "flat_symbolic")
        self.assertEqual(config.ngu.encoder_mode, "flat_symbolic")
        self.assertEqual(config.icm.encoder_mode, "flat_symbolic")

    def test_craftax_structured_encoder_mode_is_supported(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            encoder_mode="craftax_structured",
        )
        self.assertEqual(config.encoder_mode, "craftax_structured")

    def test_inventory_only_encoder_mode_is_supported(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            encoder_mode="inventory_only",
        )
        self.assertEqual(config.encoder_mode, "inventory_only")

    def test_inventory_only_encoder_mode_is_not_supported_for_full_craftax(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                env_id="Craftax-Symbolic-v1",
                encoder_mode="inventory_only",
            )

    def test_invalid_encoder_mode_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                encoder_mode="not_a_real_mode",
            )

    def test_intrinsic_modules_accept_their_own_encoder_modes(self):
        config = TrainConfig(
            **_base_config_kwargs(),
            encoder_mode="craftax_structured",
            rnd=RNDConfig(encoder_mode="inventory_only"),
            ngu=NGUConfig(encoder_mode="flat_symbolic"),
            icm=ICMConfig(encoder_mode="inventory_only"),
        )
        self.assertEqual(config.encoder_mode, "craftax_structured")
        self.assertEqual(config.rnd.encoder_mode, "inventory_only")
        self.assertEqual(config.ngu.encoder_mode, "flat_symbolic")
        self.assertEqual(config.icm.encoder_mode, "inventory_only")

    def test_invalid_intrinsic_encoder_mode_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                rnd=RNDConfig(encoder_mode="not_a_real_mode"),
            )

    def test_inventory_only_intrinsic_encoder_is_not_supported_for_full_craftax(self):
        with self.assertRaises(ValueError):
            TrainConfig(
                **_base_config_kwargs(),
                env_id="Craftax-Symbolic-v1",
                rnd=RNDConfig(encoder_mode="inventory_only"),
            )

    def test_use_inventory_only_flags_are_not_supported(self):
        with self.assertRaises(TypeError):
            RNDConfig(use_inventory_only=True)

        with self.assertRaises(TypeError):
            NGUConfig(use_inventory_only=True)


if __name__ == "__main__":
    unittest.main()
