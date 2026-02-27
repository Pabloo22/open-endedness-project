import unittest
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import numpy as np

from crew.main_algo.logging import (
    build_eval_log_payload,
    build_training_batch_log_payload,
    build_wandb_run_name,
    log_outer_batch_to_wandb,
)


class _FakeWandb:
    class Histogram:
        def __init__(self, values):
            self.values = np.asarray(values)

    def __init__(self):
        self.log_calls = []

    def log(self, payload, step, commit):
        self.log_calls.append(
            {
                "payload": payload,
                "step": step,
                "commit": commit,
            }
        )


class TestMainAlgoLogging(unittest.TestCase):
    def test_build_wandb_run_name_includes_intrinsic_modules(self):
        config = SimpleNamespace(
            env_id="Craftax-Classic-Symbolic-v1",
            train_seed=7,
            wandb_run_name=None,
            selected_intrinsic_modules=("rnd", "foo"),
        )
        self.assertEqual(
            build_wandb_run_name(config),
            "main_algo|Craftax-Classic-Symbolic-v1|seed7|intr:rnd+foo",
        )

    def test_build_training_batch_log_payload_splits_reward_vector_metrics(self):
        reward_names = ("extrinsic", "rnd")
        batch_metrics = {
            "run/batch_idx": jnp.array(3, dtype=jnp.int32),
            "run/total_env_steps": jnp.array(120, dtype=jnp.int32),
            "time/cumulative_wall_clock_sec": jnp.array(2.5, dtype=jnp.float32),
            "time/env_steps_per_sec": jnp.array(48.0, dtype=jnp.float32),
            "preproc/weighted_adv_mean": jnp.array(0.1, dtype=jnp.float32),
            "preproc/weighted_adv_std": jnp.array(1.2, dtype=jnp.float32),
            "ppo/total_loss": jnp.array(0.9, dtype=jnp.float32),
            "ppo/actor_loss": jnp.array(0.3, dtype=jnp.float32),
            "ppo/entropy": jnp.array(0.02, dtype=jnp.float32),
            "ppo/approx_kl": jnp.array(0.01, dtype=jnp.float32),
            "intrinsic_modules/rnd/predictor_loss": jnp.array(0.4, dtype=jnp.float32),
            "curriculum/pred_score_mean": jnp.array(0.8, dtype=jnp.float32),
            "curriculum/predictor_loss": jnp.array(0.7, dtype=jnp.float32),
            "curriculum/alpha/entropy_mean": jnp.array(0.6, dtype=jnp.float32),
            "curriculum/score_mean": jnp.array(0.5, dtype=jnp.float32),
            "curriculum/valid_fraction_of_scores_in_batch": jnp.array(1.0, dtype=jnp.float32),
            "curriculum/completed_episodes_per_env_mean": jnp.array(2.0, dtype=jnp.float32),
            "preproc/adv_raw_mean": jnp.array([0.1, 0.2], dtype=jnp.float32),
            "preproc/adv_norm_mean": jnp.array([0.0, 0.0], dtype=jnp.float32),
            "preproc/adv_norm_std": jnp.array([1.0, 1.0], dtype=jnp.float32),
            "ppo/value_loss": jnp.array([0.4, 0.6], dtype=jnp.float32),
            "curriculum/alpha/mean_per_reward_function": jnp.array([0.9, 0.1], dtype=jnp.float32),
            "curriculum/alpha/std_per_reward_function": jnp.array([0.2, 0.2], dtype=jnp.float32),
            "curriculum/lp_per_reward_function": jnp.array([0.3, 0.4], dtype=jnp.float32),
        }

        payload = build_training_batch_log_payload(batch_metrics=batch_metrics, reward_function_names=reward_names)

        self.assertAlmostEqual(payload["ppo/value_loss/extrinsic"], 0.4, places=6)
        self.assertAlmostEqual(payload["ppo/value_loss/rnd"], 0.6, places=6)
        self.assertAlmostEqual(payload["preproc/adv_raw_mean/extrinsic"], 0.1, places=6)
        self.assertAlmostEqual(payload["preproc/adv_raw_mean/rnd"], 0.2, places=6)
        self.assertEqual(payload["run/total_env_steps"], 120)

    def test_build_eval_log_payload(self):
        eval_metrics = {
            "eval/returns": jnp.array([[[1.0, 3.0], [2.0, 4.0]]], dtype=jnp.float32),  # [A=1,B=2,E=2]
            "eval/lengths": jnp.array([[[10, 14], [12, 16]]], dtype=jnp.int32),
            "eval/achievements": jnp.array(
                [[[[True, False], [True, True]], [[False, True], [False, False]]]],
                dtype=jnp.bool_,
            ),  # [1,2,2,2]
            "eval/batch_idx": jnp.array(5, dtype=jnp.int32),
            "eval/total_steps": jnp.array(1000, dtype=jnp.int32),
        }

        payload = build_eval_log_payload(
            eval_metrics=eval_metrics,
            evaluation_alpha_labels=("ext10_rnd00",),
            achievement_names=("Achievements/a", "Achievements/b"),
        )

        self.assertEqual(payload["eval/batch_num"], 5)
        self.assertEqual(payload["eval/total_steps"], 1000)
        self.assertAlmostEqual(payload["eval/ext10_rnd00/return_mean"], 2.5, places=6)
        self.assertAlmostEqual(payload["eval/ext10_rnd00/length_median"], 13.0, places=6)
        self.assertAlmostEqual(
            payload["eval/ext10_rnd00/achievement_success_percentage/Achievements/a"],
            50.0,
            places=6,
        )
        self.assertAlmostEqual(
            payload["eval/ext10_rnd00/num_accomplished_achievements_mean"],
            1.0,
            places=6,
        )

    def test_log_outer_batch_to_wandb_logs_once_with_step_and_histogram(self):
        fake_wandb = _FakeWandb()
        config = SimpleNamespace(
            selected_intrinsic_modules=("rnd",),
            evaluation_alpha_labels=("ext10_rnd00",),
        )
        batch_metrics = {
            "run/batch_idx": jnp.array(1, dtype=jnp.int32),
            "run/total_env_steps": jnp.array(128, dtype=jnp.int32),
            "time/cumulative_wall_clock_sec": jnp.array(1.0, dtype=jnp.float32),
            "time/env_steps_per_sec": jnp.array(128.0, dtype=jnp.float32),
            "preproc/weighted_adv_mean": jnp.array(0.1, dtype=jnp.float32),
            "preproc/weighted_adv_std": jnp.array(0.2, dtype=jnp.float32),
            "ppo/total_loss": jnp.array(0.3, dtype=jnp.float32),
            "ppo/actor_loss": jnp.array(0.4, dtype=jnp.float32),
            "ppo/entropy": jnp.array(0.5, dtype=jnp.float32),
            "ppo/approx_kl": jnp.array(0.6, dtype=jnp.float32),
            "intrinsic_modules/rnd/predictor_loss": jnp.array(0.7, dtype=jnp.float32),
            "curriculum/pred_score_mean": jnp.array(0.8, dtype=jnp.float32),
            "curriculum/predictor_loss": jnp.array(0.9, dtype=jnp.float32),
            "curriculum/alpha/entropy_mean": jnp.array(1.0, dtype=jnp.float32),
            "curriculum/score_mean": jnp.array(1.1, dtype=jnp.float32),
            "curriculum/valid_fraction_of_scores_in_batch": jnp.array(1.0, dtype=jnp.float32),
            "curriculum/completed_episodes_per_env_mean": jnp.array(2.0, dtype=jnp.float32),
            "preproc/adv_raw_mean": jnp.array([0.1, 0.2], dtype=jnp.float32),
            "preproc/adv_norm_mean": jnp.array([0.0, 0.0], dtype=jnp.float32),
            "preproc/adv_norm_std": jnp.array([1.0, 1.0], dtype=jnp.float32),
            "ppo/value_loss": jnp.array([0.2, 0.4], dtype=jnp.float32),
            "curriculum/alpha/mean_per_reward_function": jnp.array([0.7, 0.3], dtype=jnp.float32),
            "curriculum/alpha/std_per_reward_function": jnp.array([0.1, 0.1], dtype=jnp.float32),
            "curriculum/lp_per_reward_function": jnp.array([0.3, 0.6], dtype=jnp.float32),
            "curriculum/alpha/extrinsic_weight_per_env": jnp.array([0.4, 0.5, 0.6], dtype=jnp.float32),
        }
        eval_metrics = {
            "eval/returns": jnp.array([[[1.0]]], dtype=jnp.float32),
            "eval/lengths": jnp.array([[[10]]], dtype=jnp.int32),
            "eval/achievements": jnp.array([[[[True]]]], dtype=jnp.bool_),
            "eval/batch_idx": jnp.array(1, dtype=jnp.int32),
            "eval/total_steps": jnp.array(128, dtype=jnp.int32),
        }

        with mock.patch("crew.main_algo.logging.wandb", fake_wandb):
            log_outer_batch_to_wandb(
                run=object(),
                batch_metrics=batch_metrics,
                config=config,
                eval_metrics=eval_metrics,
                achievement_names=("Achievements/a",),
            )

        self.assertEqual(len(fake_wandb.log_calls), 1)
        call = fake_wandb.log_calls[0]
        self.assertEqual(call["step"], 128)
        self.assertTrue(call["commit"])
        self.assertIn("curriculum/alpha/extrinsic_weight_histogram", call["payload"])
        self.assertIsInstance(
            call["payload"]["curriculum/alpha/extrinsic_weight_histogram"],
            _FakeWandb.Histogram,
        )
        self.assertIn("eval/ext10_rnd00/return_mean", call["payload"])


if __name__ == "__main__":
    unittest.main()
