from dataclasses import asdict

import jax.numpy as jnp
import wandb

# -----------------------------------------------------------------------------------------------


def generate_run_name(algorithm_name, config, prefix=""):
    run_name = prefix + f"|{config.env_id}|" + algorithm_name + f"seed{config.train_seed}||"
    return run_name


# -----------------------------------------------------------------------------------------------


def wandb_log_training_metrics(metrics, config, run_name, project_name="open_endedness", num_final_episodes_for_evaluating_performance=None, extra_batch_metrics=None, tags=None):
    run = None

    try:
        run = wandb.init(project=project_name, name=run_name, tags=tags, config=asdict(config))
        # setup
        num_batches = metrics["total_loss"].shape[0]
        eval_returns_data = metrics["eval/returns"]
        eval_lengths_data = metrics["eval/lengths"]
        if num_final_episodes_for_evaluating_performance is None:
            final_k_episodes = eval_returns_data.shape[2]  # evaluate over all episodes
        else:
            final_k_episodes = min(num_final_episodes_for_evaluating_performance, eval_returns_data.shape[2])

        #### log plots of metrics during training ####

        total_loss = metrics["total_loss"]
        actor_loss = metrics["actor_loss"]
        value_loss = metrics["value_loss"]
        entropy = metrics["entropy"]
        kl = metrics["kl"]

        eval_returns_mean = eval_returns_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_lengths_mean = eval_lengths_data.mean(1)[:, -final_k_episodes:].mean(1)
        eval_lengths_20percentile = jnp.percentile(eval_lengths_data, q=20, axis=1)[:, -final_k_episodes:].mean(1)
        eval_lengths_40percentile = jnp.percentile(eval_lengths_data, q=40, axis=1)[:, -final_k_episodes:].mean(1)
        eval_lengths_60percentile = jnp.percentile(eval_lengths_data, q=60, axis=1)[:, -final_k_episodes:].mean(1)
        eval_lengths_80percentile = jnp.percentile(eval_lengths_data, q=80, axis=1)[:, -final_k_episodes:].mean(1)

        for i in range(num_batches):
            batch_logs = {
                # Training losses
                "training/loss/total": total_loss[i],
                "training/loss/actor": actor_loss[i],
                "training/loss/value": value_loss[i],
                "training/loss/entropy": entropy[i],
                "training/kl": kl[i],
                # evaluation metrics
                "eval/returns/mean": eval_returns_mean[i],
                "eval/episode_length/mean": eval_lengths_mean[i],
                "eval/episode_length/p20": eval_lengths_20percentile[i],
                "eval/episode_length/p40": eval_lengths_40percentile[i],
                "eval/episode_length/p60": eval_lengths_60percentile[i],
                "eval/episode_length/p80": eval_lengths_80percentile[i],
            }

            if extra_batch_metrics:
                for k, arr in extra_batch_metrics.items():
                    batch_logs[k] = arr[i]

            wandb.log(batch_logs, step=i + 1, commit=False)

        ### Log data of performance across episodes within a lifetime ###

        # configuration
        batches_for_evaluation = (num_batches - min(15, num_batches // 2), num_batches)
        num_episodes = eval_returns_data.shape[2]

        # metrics to plot
        eval_episodes_lengths_mean = eval_lengths_data.mean(1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_20percentile = jnp.percentile(eval_lengths_data, q=20, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_40percentile = jnp.percentile(eval_lengths_data, q=40, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_60percentile = jnp.percentile(eval_lengths_data, q=60, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)
        eval_episodes_lengths_80percentile = jnp.percentile(eval_lengths_data, q=80, axis=1)[batches_for_evaluation[0] : batches_for_evaluation[1]].mean(0)

        for i in range(num_episodes):
            wandb.log(
                {
                    "episode/length/mean": eval_episodes_lengths_mean[i],
                    "episode/length/p20": eval_episodes_lengths_20percentile[i],
                    "episode/length/p40": eval_episodes_lengths_40percentile[i],
                    "episode/length/p60": eval_episodes_lengths_60percentile[i],
                    "episode/length/p80": eval_episodes_lengths_80percentile[i],
                    "episode_number": i + 1,
                },
                step=num_batches + 2 + i,
                commit=False,
            )

        wandb.log({}, commit=True)
    except Exception as e:
        print(f"Error while logging training metrics to wandb: {e}")
    finally:
        if run is not None:
            run.finish()
