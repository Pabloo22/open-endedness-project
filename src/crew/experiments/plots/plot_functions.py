import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap


def plot_1_heatmaps(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    plt.style.use("ggplot")
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.875 + step / 2, step), 3)
    grid_dim = len(grid_indices)
    max_score = len(achievement_filter.split("+")) if achievement_filter else 1.0

    def create_grid(run_type):
        grid = np.full((grid_dim, grid_dim), np.nan)
        mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)

        subset = df[df["run_type"] == run_type]
        if subset.empty:
            return grid, mask_invalid

        grouped = subset.groupby(["icm_weight", "rnd_weight"])["final_performance"].mean().reset_index()

        for i, icm in enumerate(grid_indices):
            for j, rnd in enumerate(grid_indices):
                # allow a bit of floating point tolerance for the max sum which is 0.9
                if round(icm + rnd, 3) > 0.9:
                    mask_invalid[i, j] = True
                else:
                    match = grouped[
                        (np.isclose(grouped["icm_weight"], icm, atol=1e-3))
                        & (np.isclose(grouped["rnd_weight"], rnd, atol=1e-3))
                    ]
                    if not match.empty:
                        grid[i, j] = match["final_performance"].values[0]
        return grid, mask_invalid

    grid_fixed, mask_invalid_fixed = create_grid("baseline")
    grid_curr, mask_invalid_curr = create_grid("curriculum")

    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad("white")

    plot_configs = [
        (grid_fixed, mask_invalid_fixed, f"plot_1_heatmap_fixed_{achievement_filter}.pdf"),
        (grid_curr, mask_invalid_curr, f"plot_1_heatmap_curriculum_{achievement_filter}.pdf"),
    ]

    for grid, mask, filename in plot_configs:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_facecolor("lightgray")

        sns.heatmap(
            grid,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            vmin=0.0,
            vmax=max_score,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            cbar_kws={"label": "Mean Final Return"},
            ax=ax,
        )

        sns.heatmap(
            mask,
            mask=~mask,
            cmap=ListedColormap(["black"]),
            cbar=False,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            ax=ax,
        )

        ax.set_xlabel("RND Weight")
        ax.set_ylabel("ICM Weight")
        ax.invert_yaxis()

        plt.tight_layout()

        # Save the figure as a PDF
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")

        plt.show()


def plot_2_learning_curves(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe"):
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))

    def get_best_run(search_df):
        if search_df.empty:
            return None, None

        # 1. Compute mean final performance for each configuration
        mean_perfs = search_df.groupby(["icm_weight", "rnd_weight"])["final_performance"].mean().reset_index()
        if mean_perfs.empty:
            return None, None

        max_perf = mean_perfs["final_performance"].max()

        # 2. Filter candidates within a 0.02 tolerance of the maximum performance
        candidates = mean_perfs[mean_perfs["final_performance"] >= max_perf - 0.02]

        best_metric = -np.inf
        best_weights = None

        # 3. For each candidate, calculate area under curve surrogate (mean of history) to find fastest learner
        for _, row in candidates.iterrows():
            icm, rnd = row["icm_weight"], row["rnd_weight"]
            cand_df = search_df[(search_df["icm_weight"] == icm) & (search_df["rnd_weight"] == rnd)]

            histories = pd.concat(cand_df["history"].tolist())
            auc_surrogate = histories["standardized_return_mean"].mean()

            if auc_surrogate > best_metric:
                best_metric = auc_surrogate
                best_weights = (icm, rnd)

        if best_weights:
            best_df = search_df[
                (search_df["icm_weight"] == best_weights[0]) & (search_df["rnd_weight"] == best_weights[1])
            ]
            return best_weights, best_df
        return None, None

    ext_only_df = df[(df["run_type"] == "baseline") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]

    fixed_search_df = df[(df["run_type"] == "baseline") & ~((df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0))]
    best_fixed_weights, best_fixed_df = get_best_run(fixed_search_df)

    # For curriculum, we pick the best alpha value for each timestep individually for each seed.
    # With those values (one per seed), the plot_curve function will compute the mean, min, and max.
    curr_search_df = df[df["run_type"] == "curriculum"]
    best_curr_df = pd.DataFrame()
    if not curr_search_df.empty:
        # Explode histories to get all time steps
        all_curr_histories = []
        for _, row in curr_search_df.iterrows():
            hist = row["history"].copy()
            hist["seed"] = row["seed"]
            hist["alpha_id"] = f"{row['icm_weight']}_{row['rnd_weight']}"
            all_curr_histories.append(hist)

        curr_histories_df = pd.concat(all_curr_histories)

        # For each seed and for each timestep, take the maximum return across all alphas
        best_per_step_seed = (
            curr_histories_df.groupby(["eval/total_steps", "seed"])["standardized_return_mean"].max().reset_index()
        )

        # Package this back into a format that plot_curve expects (a df with a 'history' column holding dfs)
        best_curr_run_data = []
        for seed in best_per_step_seed["seed"].unique():
            seed_hist = best_per_step_seed[best_per_step_seed["seed"] == seed][
                ["eval/total_steps", "standardized_return_mean"]
            ]
            best_curr_run_data.append({"history": seed_hist})

        best_curr_df = pd.DataFrame(best_curr_run_data)

    def plot_curve(subset_df, label, color):
        if subset_df is None or subset_df.empty:
            return

        histories = pd.concat(subset_df["history"].tolist())
        histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
        stats = (
            histories.groupby("eval/total_steps")["standardized_return_mean"].agg(["mean", "min", "max"]).reset_index()
        )

        plt.plot(stats["eval/total_steps"], stats["mean"], label=label, color=color)
        plt.fill_between(
            stats["eval/total_steps"],
            stats["min"],
            stats["max"],
            color=color,
            alpha=0.2,
        )

    plot_curve(ext_only_df, "Extrinsic Only", "blue")

    if best_fixed_weights:
        plot_curve(
            best_fixed_df, f"Best Fixed Weights (ICM={best_fixed_weights[0]}, RND={best_fixed_weights[1]})", "orange"
        )

    if not best_curr_df.empty:
        plot_curve(best_curr_df, "Curriculum Method (Ours)", "green")

    plt.xlabel("Total Environment Steps")
    plt.ylabel("Extrinsic Return (Mean)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure as a PDF
    save_path = os.path.join(save_dir, f"plot_2_learning_curves_{achievement_filter}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved Plot 2 to {save_path}")

    plt.show()
