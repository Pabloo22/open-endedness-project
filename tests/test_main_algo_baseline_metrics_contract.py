import unittest

import jax

from curemix.main_algo.baseline_main_loop import full_training_baseline
from curemix.main_algo.config import TrainConfig
from curemix.main_algo.setups import set_up_for_training


class TestMainAlgoBaselineMetricsContract(unittest.TestCase):
    def test_full_training_baseline_metrics_keys_match_contract(self):
        expected_training_keys = {
            "preproc/adv_raw_mean",
            "preproc/adv_norm_mean",
            "preproc/adv_norm_std",
            "preproc/weighted_adv_mean",
            "preproc/weighted_adv_std",
            "ppo/total_loss",
            "ppo/value_loss",
            "ppo/actor_loss",
            "ppo/entropy",
            "ppo/approx_kl",
        }
        expected_final_keys = {
            *expected_training_keys,
            "run/batch_idx",
            "run/total_env_steps",
            "time/cumulative_wall_clock_sec",
            "time/env_steps_per_sec",
            "eval/returns",
            "eval/lengths",
            "eval/achievements",
            "eval/batch_idx",
            "eval/total_steps",
            "eval/achievement_names",
            "eval/alphas",
        }

        config = TrainConfig(
            training_mode="baseline",
            selected_intrinsic_modules=(),
            train_seed=0,
            total_timesteps=16,
            env_id="Craftax-Classic-Symbolic-v1",
            num_envs_per_batch=4,
            num_steps_per_env=4,
            num_steps_per_update=4,
            update_epochs=1,
            num_minibatches=1,
            past_context_length=4,
            subsequence_length_in_loss_calculation=4,
            num_transformer_blocks=1,
            transformer_hidden_states_dim=16,
            qkv_features=16,
            num_attn_heads=2,
            head_hidden_dim=16,
            obs_emb_dim=32,
            eval_every_n_batches=1,
            eval_num_envs=2,
            eval_num_episodes=1,
            enable_wandb=False,
        )
        (
            rng,
            env,
            env_params,
            agent_train_state,
            reward_normalization_stats,
            intrinsic_modules,
            intrinsic_states,
            _curriculum_state,
        ) = set_up_for_training(config)
        out = jax.block_until_ready(
            full_training_baseline(
                rng=rng,
                agent_train_state=agent_train_state,
                reward_normalization_stats=reward_normalization_stats,
                intrinsic_states=intrinsic_states,
                env=env,
                env_params=env_params,
                intrinsic_modules=intrinsic_modules,
                config=config,
            )
        )
        metrics = out["metrics"]
        self.assertEqual(set(metrics.keys()), expected_final_keys)
        self.assertEqual(len(metrics), len(expected_final_keys))


if __name__ == "__main__":
    unittest.main()
