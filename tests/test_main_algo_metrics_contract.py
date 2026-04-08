import unittest

import jax

from curemix.main_algo.config import CurriculumConfig, RNDConfig, TrainConfig
from curemix.main_algo.main_loop import full_training
from curemix.main_algo.setups import set_up_for_training


class TestMainAlgoMetricsContract(unittest.TestCase):
    def test_full_training_metrics_keys_match_contract(self):
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
            "intrinsic_modules/rnd/predictor_loss",
            "curriculum/pred_score_mean",
            "curriculum/predictor_loss",
            "curriculum/alpha/mean_per_reward_function",
            "curriculum/alpha/std_per_reward_function",
            "curriculum/alpha/per_env",
            "curriculum/alpha/entropy_mean",
            "curriculum/lp_per_reward_function",
            "curriculum/score_mean",
            "curriculum/valid_fraction_of_scores_in_batch",
            "curriculum/completed_episodes_per_env_mean",
            "curriculum/alpha/extrinsic_weight_per_env",
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
            train_seed=0,
            total_timesteps=32,
            env_id="Craftax-Classic-Symbolic-v1",
            num_envs_per_batch=4,
            num_steps_per_env=8,
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
            curriculum=CurriculumConfig(
                replay_buffer_num_batches=2,
                predictor_num_minibatches=2,
                predictor_update_epochs=1,
                predictor_hidden_dim=16,
                importance_num_candidates_multiplier=2,
                min_batches_for_predictor_sampling=1,
            ),
            rnd=RNDConfig(
                predictor_num_minibatches=4,
                num_chunks_in_rewards_computation=4,
                predictor_update_epochs=1,
                head_hidden_dim=32,
                output_embedding_dim=32,
            ),
        )
        (
            rng,
            env,
            env_params,
            agent_train_state,
            reward_normalization_stats,
            intrinsic_modules,
            intrinsic_states,
            curriculum_state,
        ) = set_up_for_training(config)
        out = jax.block_until_ready(
            full_training(
                rng=rng,
                agent_train_state=agent_train_state,
                reward_normalization_stats=reward_normalization_stats,
                intrinsic_states=intrinsic_states,
                curriculum_state=curriculum_state,
                env=env,
                env_params=env_params,
                intrinsic_modules=intrinsic_modules,
                config=config,
            )
        )
        metrics = out["metrics"]
        self.assertEqual(set(metrics.keys()), expected_final_keys)
        self.assertEqual(len(metrics), len(expected_final_keys))
        self.assertEqual(
            metrics["curriculum/alpha/per_env"].shape,
            (
                config.num_batches_of_envs,
                config.num_envs_per_batch,
                config.num_reward_functions,
            ),
        )


if __name__ == "__main__":
    unittest.main()
