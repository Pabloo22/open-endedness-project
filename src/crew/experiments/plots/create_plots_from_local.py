"""Creates the plots for the paper using local data.

Available keys in 'metrics' for the currriculum run:
 - 'curriculum/alpha/entropy_mean': array shape (239,)
 - 'curriculum/alpha/extrinsic_weight_per_env': array shape (239, 1024)
 - 'curriculum/alpha/mean_per_reward_function': array shape (239, 3)
 - 'curriculum/alpha/per_env': array shape (239, 1024, 3)
 - 'curriculum/alpha/std_per_reward_function': array shape (239, 3)
 - 'curriculum/completed_episodes_per_env_mean': array shape (239,)
 - 'curriculum/lp_per_reward_function': array shape (239, 3)
 - 'curriculum/pred_score_mean': array shape (239,)
 - 'curriculum/predictor_loss': array shape (239,)
 - 'curriculum/score_mean': array shape (239,)
 - 'curriculum/valid_fraction_of_scores_in_batch': array shape (239,)
 - 'eval/achievement_names': <class 'list'>
 - 'eval/achievements': array shape (48, 36, 32, 8, 22)
 - 'eval/alphas': array shape (36, 3)
 - 'eval/batch_idx': array shape (48,)
 - 'eval/lengths': array shape (48, 36, 32, 8)
 - 'eval/returns': array shape (48, 36, 32, 8)
 - 'eval/total_steps': array shape (48,)
 - 'intrinsic_modules/icm/forward_loss': array shape (239,)
 - 'intrinsic_modules/icm/inverse_loss': array shape (239,)
 - 'intrinsic_modules/icm/loss': array shape (239,)
 - 'intrinsic_modules/rnd/predictor_loss': array shape (239,)
 - 'ppo/actor_loss': array shape (239,)
 - 'ppo/approx_kl': array shape (239,)
 - 'ppo/entropy': array shape (239,)
 - 'ppo/total_loss': array shape (239,)
 - 'ppo/value_loss': array shape (239, 3)
 - 'preproc/adv_norm_mean': array shape (239, 3)
 - 'preproc/adv_norm_std': array shape (239, 3)
 - 'preproc/adv_raw_mean': array shape (239, 3)
 - 'preproc/weighted_adv_mean': array shape (239,)
 - 'preproc/weighted_adv_std': array shape (239,)
 - 'run/batch_idx': array shape (239,)
 - 'run/total_env_steps': array shape (239,)
 - 'time/cumulative_wall_clock_sec': array shape (239,)
 - 'time/env_steps_per_sec': array shape (239,)
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import orbax.checkpoint as ocp
from crew.experiments.plots.plot_functions import (
    plot_1_heatmaps,
    plot_2_learning_curves,
    plot_3_curriculum_adaptation,
    plot_5_heatmaps,
    plot_6_learning_curves,
)

# ==========================================
# CONFIGURATION
# ==========================================
ARTIFACTS_DIR = "artifacts/training_results"
ACHIEVEMENT_FILTER = "place_furnace+make_iron_pickaxe"


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_weights_from_dirname(dirname):
    """
    Parses a directory name like 'icm0p125+rnd0' into floats.
    Returns: icm_weight, rnd_weight
    """
    pattern = r"^icm(.*)\+rnd(.*?)$"
    match = re.match(pattern, dirname)
    if not match:
        return 0.0, 0.0

    icm_str, rnd_str = match.groups()
    icm_weight = float(icm_str.replace("p", ".")) if icm_str else 0.0
    rnd_weight = float(rnd_str.replace("p", ".")) if rnd_str else 0.0

    return icm_weight, rnd_weight


def load_local_orbax_data(base_dir, achievement):
    """
    Walks the local directory structure, loads Orbax checkpoints,
    processes raw multi-dimensional JAX arrays, and returns a pandas DataFrame.
    """
    target_dir = Path(base_dir) / achievement
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist.")
        return pd.DataFrame()

    checkpointer = ocp.PyTreeCheckpointer()
    data = []

    print(f"Scanning local directory: {target_dir}...")

    for run_type_dir in target_dir.iterdir():
        if not run_type_dir.is_dir():
            continue
        run_type = run_type_dir.name

        for weights_dir in run_type_dir.iterdir():
            if not weights_dir.is_dir():
                continue
            icm_weight, rnd_weight = parse_weights_from_dirname(weights_dir.name)

            for seed_dir in weights_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                    continue

                seed = int(seed_dir.name.replace("seed", ""))
                checkpoint_path = seed_dir.resolve()

                print(f"Loading: {run_type} | ICM: {icm_weight} | RND: {rnd_weight} | Seed: {seed}")

                try:
                    restored_pytree = checkpointer.restore(os.path.abspath(checkpoint_path))
                    metrics = restored_pytree.get("metrics", {})

                    if not metrics:
                        print(f"  -> Warning: No metrics found in {checkpoint_path}")
                        continue

                    # 1. Fetch raw arrays and convert to standard NumPy to avoid JAX device quirks
                    if "eval/total_steps" not in metrics or "eval/returns" not in metrics:
                        print(f"  -> Warning: Required metric arrays missing in {checkpoint_path}")
                        continue

                    steps_array = np.asarray(metrics["eval/total_steps"])
                    raw_returns = np.asarray(metrics["eval/returns"])  # Shape: (T, Alphas, Envs, Episodes)

                    # Also fetch achievements if available
                    raw_achs = None
                    if "eval/achievements" in metrics:
                        raw_achs = np.asarray(metrics["eval/achievements"])

                    # 2. Process the 4D returns array into a 1D array of means
                    if run_type == "baseline":
                        # Average across Alphas (1), Envs (256), and Episodes (1)
                        returns_array = np.mean(raw_returns, axis=(1, 2, 3))

                        hist_dict = {
                            "eval/total_steps": steps_array,
                            "standardized_return_mean": returns_array,
                        }

                        final_achievements = np.nan
                        if raw_achs is not None:
                            # sum over achievements (axis=-1), then avg over alphas, envs, episodes
                            achs_array = np.mean(np.sum(raw_achs, axis=-1), axis=(1, 2, 3))
                            hist_dict["achievements_mean"] = achs_array
                            final_achievements = achs_array[-1]

                        # find min len across all arrays to handle mismatch
                        min_len = min(len(v) for v in hist_dict.values())

                        hist_dict = {k: v[:min_len] for k, v in hist_dict.items()}

                        history = pd.DataFrame(hist_dict)

                        history = history.dropna(subset=["eval/total_steps", "standardized_return_mean"])

                        if history.empty:
                            print(f"  -> Warning: Metric arrays were empty after cleaning.")
                            continue

                        final_performance = history["standardized_return_mean"].iloc[-1]

                        data.append(
                            {
                                "run_id": f"{run_type}_{weights_dir.name}_{seed_dir.name}",
                                "run_type": run_type,
                                "icm_weight": icm_weight,
                                "rnd_weight": rnd_weight,
                                "extrinsic_weight": round(1.0 - icm_weight - rnd_weight, 2),
                                "seed": seed,
                                "final_performance": final_performance,
                                "final_achievements": final_achievements,
                                "history": history,
                            }
                        )
                    else:
                        # Extract alpha history for plot 3
                        alpha_history = pd.DataFrame()
                        if "curriculum/alpha/mean_per_reward_function" in metrics and "run/total_env_steps" in metrics:
                            alpha_means = np.asarray(metrics["curriculum/alpha/mean_per_reward_function"])
                            run_steps = np.asarray(metrics["run/total_env_steps"])
                            min_len_alpha = min(len(run_steps), len(alpha_means))
                            alpha_history = pd.DataFrame(
                                {
                                    "run/total_env_steps": run_steps[:min_len_alpha],
                                    "alpha_ext": alpha_means[:min_len_alpha, 0],
                                    "alpha_icm": alpha_means[:min_len_alpha, 1],
                                    "alpha_rnd": alpha_means[:min_len_alpha, 2],
                                }
                            )

                        # For curriculum, run provides results for multiple alphas
                        alphas = np.asarray(metrics.get("eval/alphas", [[1.0, 0.0, 0.0]]))
                        for alpha_idx, alpha_vals in enumerate(alphas):
                            cur_icm_weight = round(float(alpha_vals[1]), 3)
                            cur_rnd_weight = round(float(alpha_vals[2]), 3)
                            cur_ext_weight = round(float(alpha_vals[0]), 3)

                            # Average only across Envs and Episodes for the specific alpha
                            returns_array = np.mean(raw_returns[:, alpha_idx, :, :], axis=(1, 2))

                            hist_dict = {
                                "eval/total_steps": steps_array,
                                "standardized_return_mean": returns_array,
                            }

                            final_achievements = np.nan
                            if raw_achs is not None:
                                # sum over achievements (axis=-1), then avg over envs (axis=1), episodes (axis=2)
                                achs_array = np.mean(np.sum(raw_achs[:, alpha_idx, :, :], axis=-1), axis=(1, 2))
                                hist_dict["achievements_mean"] = achs_array
                                final_achievements = achs_array[-1]

                            min_len = min(len(v) for v in hist_dict.values())
                            hist_dict = {k: v[:min_len] for k, v in hist_dict.items()}

                            history = pd.DataFrame(hist_dict)

                            history = history.dropna(subset=["eval/total_steps", "standardized_return_mean"])

                            if history.empty:
                                continue

                            final_performance = history["standardized_return_mean"].iloc[-1]

                            data.append(
                                {
                                    "run_id": f"{run_type}_{weights_dir.name}_{seed_dir.name}_alpha{alpha_idx}",
                                    "run_type": run_type,
                                    "icm_weight": cur_icm_weight,
                                    "rnd_weight": cur_rnd_weight,
                                    "extrinsic_weight": cur_ext_weight,
                                    "seed": seed,
                                    "final_performance": final_performance,
                                    "final_achievements": final_achievements,
                                    "history": history,
                                    "alpha_history": alpha_history,
                                }
                            )

                except Exception as e:
                    print(f"  -> Failed to load {checkpoint_path}: {e}")

    return pd.DataFrame(data)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Fetch data locally instead of from W&B
    df = load_local_orbax_data(ARTIFACTS_DIR, ACHIEVEMENT_FILTER)

    if not df.empty:
        print(f"\nSuccessfully loaded {len(df)} local runs. Ready to plot!")

        # 2. You can now pass this `df` directly to the plotting functions from the previous script!
        plot_1_heatmaps(df, "images", achievement_filter=ACHIEVEMENT_FILTER)
        plot_2_learning_curves(df, "images", achievement_filter=ACHIEVEMENT_FILTER)
        plot_3_curriculum_adaptation(
            df,
            "images",
            achievement_filter=ACHIEVEMENT_FILTER,
            include_baseline_performance=True,
        )
        plot_5_heatmaps(df, "images", achievement_filter=ACHIEVEMENT_FILTER)
        plot_6_learning_curves(df, "images", achievement_filter=ACHIEVEMENT_FILTER)

    else:
        print("No valid local data found.")
    # import os
    # import orbax.checkpoint as ocp

    # # Pointing exactly to the run that failed
    # path = "/cs/student/msc/ml/2025/parinofe/open-endedness-project/artifacts/training_results/place_furnace+make_iron_pickaxe/curriculum/icm+rnd/seed1"

    # checkpointer = ocp.PyTreeCheckpointer()
    # data = checkpointer.restore(os.path.abspath(path))
    # metrics = data.get("metrics", {})

    # print("Available keys in 'metrics':")
    # for key, value in metrics.items():
    #     if hasattr(value, "shape"):
    #         print(f" - '{key}': array shape {value.shape}")
    #     elif isinstance(value, dict):
    #         print(f" - '{key}': nested dictionary with keys {list(value.keys())}")
    #     else:
    #         print(f" - '{key}': {type(value)}")
