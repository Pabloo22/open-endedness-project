import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_1_heatmaps(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    plt.style.use("ggplot")
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.5 + step / 2, step), 3)
    grid_dim = len(grid_indices)

    def create_grid(run_type):
        grid = np.full((grid_dim, grid_dim), np.nan)
        mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)

        subset = df[df["run_type"] == run_type]
        if subset.empty:
            return grid, mask_invalid

        grouped = subset.groupby(["icm_weight", "rnd_weight"])["final_performance"].mean().reset_index()

        for i, icm in enumerate(grid_indices):
            for j, rnd in enumerate(grid_indices):
                # allow a bit of floating point tolerance for the max sum which is 1.0
                if round(icm + rnd, 3) > 1.0:
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
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            grid,
            mask=mask,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            cbar_kws={"label": "Mean Final Return"},
        )

        ax = plt.gca()
        ax.set_facecolor("black")
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

    ext_only_df = df[(df["run_type"] == "baseline") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]
    fixed_search_df = df[(df["run_type"] == "baseline") & ~((df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0))]
    best_fixed_weights = None

    if not fixed_search_df.empty:
        mean_perfs = fixed_search_df.groupby(["icm_weight", "rnd_weight"])["final_performance"].mean()
        if not mean_perfs.empty:
            best_icm, best_rnd = mean_perfs.idxmax()
            best_fixed_df = fixed_search_df[
                (fixed_search_df["icm_weight"] == best_icm) & (fixed_search_df["rnd_weight"] == best_rnd)
            ]
            best_fixed_weights = (best_icm, best_rnd)

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

    plot_curve(ext_only_df, "Extrinsic Only (ICM=0.0, RND=0.0)", "blue")

    if best_fixed_weights:
        plot_curve(
            best_fixed_df, f"Best Fixed Weights (ICM={best_fixed_weights[0]}, RND={best_fixed_weights[1]})", "orange"
        )

    plt.xlabel("Total Environment Steps")
    plt.ylabel("Extrinsic Return (Mean)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure as a PDF
    save_path = os.path.join(save_dir, f"plot_2_learning_curves_{achievement_filter}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved Plot 2 to {save_path}")

    plt.show()
