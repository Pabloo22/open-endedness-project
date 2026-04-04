import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from adjustText import adjust_text


# Opacity used for uncertainty/shaded regions in line plots.
SHADED_REGION_ALPHA = 0.10

# Z value for two-sided 95% confidence intervals.
CI_95_Z = 1.96

# Figure size used for line plots
LINE_PLOT_FIGSIZE = (9, 6)


def _plot_heatmaps_base(
    df,
    save_dir,
    metric_col,
    prefix,
    cbar_label,
    max_score,
    achievement_filter="place_furnace+make_iron_pickaxe",
    grid_size=8,
):
    plt.style.use("ggplot")
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.875 + step / 2, step), 3)
    grid_dim = len(grid_indices)

    def create_grid(run_type):
        grid = np.full((grid_dim, grid_dim), np.nan)
        mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)

        subset = df[df["run_type"] == run_type]
        if subset.empty or metric_col not in subset.columns:
            return grid, mask_invalid

        grouped = subset.groupby(["icm_weight", "rnd_weight"])[metric_col].mean().reset_index()

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
                        grid[i, j] = match[metric_col].values[0]
        return grid, mask_invalid

    grid_fixed, mask_invalid_fixed = create_grid("baseline")
    grid_curr, mask_invalid_curr = create_grid("curriculum")

    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad("white")

    plot_configs = [
        (grid_fixed, mask_invalid_fixed, f"{prefix}_heatmap_fixed_{achievement_filter}.pdf"),
        (grid_curr, mask_invalid_curr, f"{prefix}_heatmap_curriculum_{achievement_filter}.pdf"),
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
            cbar_kws={"label": cbar_label},
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


def plot_1_heatmaps(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    max_score = len(achievement_filter.split("+")) if achievement_filter else 1.0
    _plot_heatmaps_base(
        df,
        save_dir,
        metric_col="final_performance",
        prefix="plot_1",
        cbar_label="Mean Final Return",
        max_score=max_score,
        achievement_filter=achievement_filter,
        grid_size=grid_size,
    )


def plot_5_heatmaps(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    max_score = 22.0  # Assuming 22 total achievements possible in Craftax classic
    _plot_heatmaps_base(
        df,
        save_dir,
        metric_col="final_achievements",
        prefix="plot_5",
        cbar_label="Mean Final Achievements",
        max_score=max_score,
        achievement_filter=achievement_filter,
        grid_size=grid_size,
    )


def _plot_learning_curves_base(df, save_dir, achievement_filter, metric_col, prefix, ylabel):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    # we use the metric_col for everything instead of standardized_return_mean and final_performance
    final_metric_col = "final_performance" if metric_col == "standardized_return_mean" else "final_achievements"

    def get_best_run(search_df):
        if search_df.empty:
            return None, None

        # 1. Compute mean final performance for each configuration
        mean_perfs = search_df.groupby(["icm_weight", "rnd_weight"])[final_metric_col].mean().reset_index()
        if mean_perfs.empty:
            return None, None

        max_perf = mean_perfs[final_metric_col].max()

        # 2. Filter candidates within a 0.02 tolerance of the maximum performance
        candidates = mean_perfs[mean_perfs[final_metric_col] >= max_perf - 0.02]

        best_metric = -np.inf
        best_weights = None

        # 3. For each candidate, calculate area under curve surrogate (mean of history) to find fastest learner
        for _, row in candidates.iterrows():
            icm, rnd = row["icm_weight"], row["rnd_weight"]
            cand_df = search_df[(search_df["icm_weight"] == icm) & (search_df["rnd_weight"] == rnd)]

            histories = pd.concat(cand_df["history"].tolist())
            if metric_col not in histories.columns:
                continue
            auc_surrogate = histories[metric_col].mean()

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
            if metric_col not in hist.columns:
                continue
            hist["seed"] = row["seed"]
            hist["alpha_id"] = f"{row['icm_weight']}_{row['rnd_weight']}"
            all_curr_histories.append(hist)

        if all_curr_histories:
            curr_histories_df = pd.concat(all_curr_histories)

            # For each seed and for each timestep, take the maximum return across all alphas
            best_per_step_seed = curr_histories_df.groupby(["eval/total_steps", "seed"])[metric_col].max().reset_index()

            # Package this back into a format that plot_curve expects (a df with a 'history' column holding dfs)
            best_curr_run_data = []
            for seed in best_per_step_seed["seed"].unique():
                seed_hist = best_per_step_seed[best_per_step_seed["seed"] == seed][["eval/total_steps", metric_col]]
                best_curr_run_data.append({"history": seed_hist})

            best_curr_df = pd.DataFrame(best_curr_run_data)

    def plot_curve(subset_df, label, color):
        if subset_df is None or subset_df.empty:
            return 0

        histories = pd.concat(subset_df["history"].tolist())
        histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
        stats = histories.groupby("eval/total_steps")[metric_col].agg(["mean", "min", "max"]).reset_index()

        ax.plot(stats["eval/total_steps"], stats["mean"], label=label, color=color)
        ax.fill_between(
            stats["eval/total_steps"],
            stats["min"],
            stats["max"],
            color=color,
            alpha=SHADED_REGION_ALPHA,
        )

        last_x = stats["eval/total_steps"].iloc[-1]
        last_y = stats["mean"].iloc[-1]
        t = ax.text(
            last_x,
            last_y,
            f"  {label}",
            color=color,
            va="center",
            ha="left",
            multialignment="left",
            fontweight="bold",
            clip_on=False,
        )
        texts_list.append(t)
        return stats["eval/total_steps"].max()

    max_steps = 0
    texts_list = []

    m = plot_curve(ext_only_df, "Extrinsic Only", "#56B4E9")
    if m:
        max_steps = max(max_steps, m)

    if best_fixed_weights:
        m = plot_curve(
            best_fixed_df, f"Best Fixed Weights\n(ICM={best_fixed_weights[0]}, RND={best_fixed_weights[1]})", "#E69F00"
        )
        if m:
            max_steps = max(max_steps, m)

    if not best_curr_df.empty:
        m = plot_curve(best_curr_df, "CuReMix (Ours)", "#009E73")
        if m:
            max_steps = max(max_steps, m)

    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel(ylabel)
    if max_steps > 0:
        ax.set_xlim(0, max_steps)
    ax.grid(True, alpha=0.3)

    if texts_list:
        # Ensure accurate multiline text extents before adjustText computes overlaps.
        fig.canvas.draw()
        adjust_text(
            texts_list,
            ax=ax,
            only_move={"text": "y"},
            ensure_inside_axes=False,
            expand=(1.01, 1.2),
            force_text=(0.1, 2.0),
        )

    # Save the figure as a PDF
    save_path = os.path.join(save_dir, f"{prefix}_learning_curves_{achievement_filter}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved {prefix.replace('_', ' ').title()} to {save_path}")

    plt.show()


def plot_2_learning_curves(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe"):
    _plot_learning_curves_base(
        df,
        save_dir,
        achievement_filter,
        metric_col="standardized_return_mean",
        prefix="plot_2",
        ylabel="Extrinsic Return (Mean)",
    )


def plot_2_b_learning_curves(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe"):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    metric_col = "standardized_return_mean"
    final_metric_col = "final_performance"
    prefix = "plot_2_b"
    ylabel = "Extrinsic Return (Mean)"

    # Compute mean final performance for each configuration
    fixed_search_df = df[(df["run_type"] == "baseline") & ~((df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0))]

    max_score = len(achievement_filter.split("+")) if achievement_filter else 1.0

    max_steps = 0
    texts_list = []

    if not fixed_search_df.empty:
        mean_perfs = fixed_search_df.groupby(["icm_weight", "rnd_weight"])[final_metric_col].mean().reset_index()
        mean_perfs = mean_perfs.sort_values(by=final_metric_col, ascending=False).reset_index(drop=True)

        # Split into 5 quintiles safely
        indices_list = np.array_split(np.arange(len(mean_perfs)), 5)
        quintiles = [mean_perfs.iloc[indices] for indices in indices_list if len(indices) > 0]

        colors = sns.color_palette("rocket", 5)
        labels = [
            "Top 0-20%\nFixed Alphas",
            "Top 20-40%\nFixed Alphas",
            "Top 40-60%\nFixed Alphas",
            "Top 60-80%\nFixed Alphas",
            "Top 80-100%\nFixed Alphas",
        ]

        for q_idx, quintile_df in enumerate(quintiles):
            if quintile_df.empty:
                continue

            q_histories = []
            for _, row in quintile_df.iterrows():
                icm, rnd = row["icm_weight"], row["rnd_weight"]
                cand_df = fixed_search_df[
                    (fixed_search_df["icm_weight"] == icm) & (fixed_search_df["rnd_weight"] == rnd)
                ]
                q_histories.extend(cand_df["history"].tolist())

            if q_histories:
                histories = pd.concat(q_histories)
                if metric_col in histories.columns:
                    histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
                    stats = (
                        histories.groupby("eval/total_steps")[metric_col]
                        .agg(mean="mean", std=lambda x: x.std(ddof=0), count="count")
                        .reset_index()
                    )

                    sem = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
                    ci_half_width = CI_95_Z * sem
                    lower_bound = np.maximum(stats["mean"] - ci_half_width, 0)
                    upper_bound = np.minimum(stats["mean"] + ci_half_width, max_score)

                    ax.plot(stats["eval/total_steps"], stats["mean"], label=labels[q_idx], color=colors[q_idx])
                    ax.fill_between(
                        stats["eval/total_steps"],
                        lower_bound,
                        upper_bound,
                        color=colors[q_idx],
                        alpha=SHADED_REGION_ALPHA,
                    )

                    last_x = stats["eval/total_steps"].iloc[-1]
                    last_y = stats["mean"].iloc[-1]
                    t = ax.text(
                        last_x,
                        last_y,
                        f"  {labels[q_idx]}",
                        color=colors[q_idx],
                        va="center",
                        ha="left",
                        multialignment="left",
                        fontweight="bold",
                        clip_on=False,
                    )
                    texts_list.append(t)
                    max_steps = max(max_steps, last_x)
    # CuReMix
    curr_search_df = df[df["run_type"] == "curriculum"]
    if not curr_search_df.empty:
        all_curr_histories = []
        for _, row in curr_search_df.iterrows():
            hist = row["history"].copy()
            if metric_col not in hist.columns:
                continue
            hist["seed"] = row["seed"]
            hist["alpha_id"] = f"{row['icm_weight']}_{row['rnd_weight']}"
            all_curr_histories.append(hist)

        if all_curr_histories:
            curr_histories_df = pd.concat(all_curr_histories)

            best_per_step_seed = curr_histories_df.groupby(["eval/total_steps", "seed"])[metric_col].max().reset_index()

            best_curr_run_data = []
            for seed in best_per_step_seed["seed"].unique():
                seed_hist = best_per_step_seed[best_per_step_seed["seed"] == seed][["eval/total_steps", metric_col]]
                best_curr_run_data.append(seed_hist)

            if best_curr_run_data:
                histories = pd.concat(best_curr_run_data)
                histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
                stats = histories.groupby("eval/total_steps")[metric_col].agg(["mean", "min", "max"]).reset_index()

                ax.plot(stats["eval/total_steps"], stats["mean"], label="CuReMix (Ours)", color="#009E73")
                ax.fill_between(
                    stats["eval/total_steps"],
                    stats["min"],
                    stats["max"],
                    color="#009E73",
                    alpha=SHADED_REGION_ALPHA,
                )
                last_x = stats["eval/total_steps"].iloc[-1]
                last_y = stats["mean"].iloc[-1]
                t = ax.text(
                    last_x,
                    last_y,
                    "  CuReMix (Ours)",
                    color="#009E73",
                    va="center",
                    ha="left",
                    multialignment="left",
                    fontweight="bold",
                    clip_on=False,
                )
                texts_list.append(t)
                max_steps = max(max_steps, last_x)

    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel(ylabel)
    if max_steps > 0:
        ax.set_xlim(0, max_steps)
    ax.grid(True, alpha=0.3)

    if texts_list:
        # Force renderer update so multiline labels have correct bounding boxes.
        fig.canvas.draw()
        adjust_text(
            texts_list,
            ax=ax,
            only_move={"text": "y"},
            ensure_inside_axes=False,
            expand=(1.01, 1.2),
            force_text=(0.1, 2.0),
        )

    save_path = os.path.join(save_dir, f"{prefix}_learning_curves_{achievement_filter}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved {prefix.replace('_', ' ').title()} to {save_path}")

    plt.show()


def plot_6_learning_curves(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe"):
    _plot_learning_curves_base(
        df,
        save_dir,
        achievement_filter,
        metric_col="achievements_mean",
        prefix="plot_6",
        ylabel="Total Achievements (Mean)",
    )


def plot_3_curriculum_adaptation(
    df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", include_baseline_performance=False
):
    plt.style.use("ggplot")
    fig, ax1 = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    curr_df = df[(df["run_type"] == "curriculum") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]

    if curr_df.empty:
        print("Warning: No curriculum data found for plot 3.")
        return

    all_alpha_histories = []
    for _, row in curr_df.iterrows():
        alpha_hist = row.get("alpha_history")
        if alpha_hist is not None and not alpha_hist.empty:
            hist_copy = alpha_hist.copy()
            hist_copy["seed"] = row["seed"]
            all_alpha_histories.append(hist_copy)

    if not all_alpha_histories:
        print("Warning: No alpha history data found for plot 3.")
        return

    combined_alpha = pd.concat(all_alpha_histories)
    combined_alpha["run/total_env_steps"] = combined_alpha["run/total_env_steps"].round(decimals=-4)

    stats = (
        combined_alpha.groupby("run/total_env_steps")[["alpha_ext", "alpha_icm", "alpha_rnd"]]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    # Make plots for each alpha
    steps = stats["run/total_env_steps"].values
    texts_list1 = []

    # Extrinsic
    ext_mean = stats["alpha_ext"]["mean"].values
    ext_min = stats["alpha_ext"]["min"].values
    ext_max = stats["alpha_ext"]["max"].values
    ax1.plot(steps, ext_mean, label="Extrinsic", color="#56B4E9")
    ax1.fill_between(steps, ext_min, ext_max, color="#56B4E9", alpha=SHADED_REGION_ALPHA)
    texts_list1.append(
        ax1.text(
            steps[-1],
            ext_mean[-1],
            "  Extrinsic",
            color="#56B4E9",
            va="center",
            ha="left",
            fontweight="bold",
            clip_on=False,
        )
    )

    # ICM
    icm_mean = stats["alpha_icm"]["mean"].values
    icm_min = stats["alpha_icm"]["min"].values
    icm_max = stats["alpha_icm"]["max"].values
    ax1.plot(steps, icm_mean, label="ICM", color="#E69F00")
    ax1.fill_between(steps, icm_min, icm_max, color="#E69F00", alpha=SHADED_REGION_ALPHA)
    texts_list1.append(
        ax1.text(
            steps[-1], icm_mean[-1], "  ICM", color="#E69F00", va="center", ha="left", fontweight="bold", clip_on=False
        )
    )

    # RND
    rnd_mean = stats["alpha_rnd"]["mean"].values
    rnd_min = stats["alpha_rnd"]["min"].values
    rnd_max = stats["alpha_rnd"]["max"].values
    ax1.plot(steps, rnd_mean, label="RND", color="#009E73")
    ax1.fill_between(steps, rnd_min, rnd_max, color="#009E73", alpha=SHADED_REGION_ALPHA)
    texts_list1.append(
        ax1.text(
            steps[-1], rnd_mean[-1], "  RND", color="#009E73", va="center", ha="left", fontweight="bold", clip_on=False
        )
    )

    ax1.set_xlabel("Total Environment Steps")
    ax1.set_ylabel("Weight")
    ax1.set_ylim(bottom=0)
    if len(steps) > 0:
        ax1.set_xlim(0, steps[-1])
    ax1.grid(True, alpha=0.3)

    if texts_list1:
        adjust_text(texts_list1, ax=ax1, only_move={"text": "y"}, ensure_inside_axes=False)

    # Save the original figure
    save_path = os.path.join(save_dir, f"plot_3_curriculum_adaptation_{achievement_filter}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved Plot 3 to {save_path}")

    # Generate the performance plot in a separate figure
    if include_baseline_performance and not curr_df.empty:
        fig2, ax_perf = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

        b_histories = pd.concat(curr_df["history"].tolist())
        b_histories["eval/total_steps"] = b_histories["eval/total_steps"].round(decimals=-4)
        b_stats = (
            b_histories.groupby("eval/total_steps")["standardized_return_mean"]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        ax_perf.plot(
            b_stats["eval/total_steps"],
            b_stats["mean"],
            color="#CC79A7",
        )
        ax_perf.fill_between(
            b_stats["eval/total_steps"],
            b_stats["min"],
            b_stats["max"],
            color="#CC79A7",
            alpha=SHADED_REGION_ALPHA,
        )

        last_x = b_stats["eval/total_steps"].iloc[-1]

        ax_perf.set_xlabel("Total Environment Steps")
        ax_perf.set_ylabel("Extrinsic Return (Mean)")
        ax_perf.set_ylim(bottom=0)
        ax_perf.set_xlim(0, last_x)
        ax_perf.grid(True, alpha=0.3)

        save_path_perf = os.path.join(save_dir, f"plot_3_curriculum_adaptation_performance_{achievement_filter}.pdf")
        fig2.savefig(save_path_perf, format="pdf", bbox_inches="tight")
        print(f"Saved Plot 3 Performance to {save_path_perf}")

    plt.show()


def plot_4_contour_overlay(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    plt.style.use("ggplot")
    import io
    import imageio
    
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.875 + step / 2, step), 3)
    grid_dim = len(grid_indices)

    max_score = len(achievement_filter.split("+")) if achievement_filter else 1.0
    metric_col = "standardized_return_mean"
    
    curr_eval_df = df[df["run_type"] == "curriculum"]
    curr_df = df[(df["run_type"] == "curriculum") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]

    max_eval_steps = 0
    if not curr_eval_df.empty:
        for _, row in curr_eval_df.iterrows():
            hist = row.get("history")
            if hist is not None and not hist.empty and "eval/total_steps" in hist.columns:
                max_eval_steps = max(max_eval_steps, hist["eval/total_steps"].max())
                
    if max_eval_steps == 0:
        print("Warning: No evaluation steps found for plot 4.")
        return
        
    percentages = np.linspace(0.0, 1.0, 21) # 0%, 5%, ..., 100%
    pdf_pcts = [0.25, 0.50, 0.75, 1.00]
    
    gif_frames = []
    
    for pct in percentages:
        target_step = max_eval_steps * pct
        
        grid_perf = np.full((grid_dim, grid_dim), np.nan)
        mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)
        
        for i, icm in enumerate(grid_indices):
            for j, rnd in enumerate(grid_indices):
                if round(icm + rnd, 3) > 0.9:
                    mask_invalid[i, j] = True
                else:
                    match = curr_eval_df[
                        (np.isclose(curr_eval_df["icm_weight"], icm, atol=1e-3))
                        & (np.isclose(curr_eval_df["rnd_weight"], rnd, atol=1e-3))
                    ]
                    if not match.empty:
                        seed_perfs = []
                        for _, row in match.iterrows():
                            hist = row.get("history")
                            if hist is not None and not hist.empty and metric_col in hist.columns:
                                valid_hists = hist[hist["eval/total_steps"] <= target_step]
                                if not valid_hists.empty:
                                    val = valid_hists.iloc[-1][metric_col]
                                    seed_perfs.append(val)
                        if seed_perfs:
                            grid_perf[i, j] = np.mean(seed_perfs)

        grid_freq = np.zeros((grid_dim, grid_dim))
        
        all_alpha_histories = []
        for _, row in curr_df.iterrows():
            alpha_hist = row.get("alpha_history")
            if alpha_hist is not None and not alpha_hist.empty and "run/total_env_steps" in alpha_hist.columns:
                valid_alpha = alpha_hist[alpha_hist["run/total_env_steps"] <= target_step]
                all_alpha_histories.append(valid_alpha)
                
        if all_alpha_histories:
            combined_alpha = pd.concat(all_alpha_histories)
            icm_vals = combined_alpha["alpha_icm"].values
            rnd_vals = combined_alpha["alpha_rnd"].values
            
            for icm_val, rnd_val in zip(icm_vals, rnd_vals):
                i_idx = np.abs(grid_indices - icm_val).argmin()
                j_idx = np.abs(grid_indices - rnd_val).argmin()
                grid_freq[i_idx, j_idx] += 1
                
            max_freq = grid_freq.max()
            if max_freq > 0:
                grid_freq = grid_freq / max_freq

        fig, ax = plt.subplots(figsize=(8, 6))
        
        X, Y = np.meshgrid(grid_indices, grid_indices)
        
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmap.set_bad("white")
        
        masked_perf = np.ma.masked_array(grid_perf, mask=mask_invalid)
        heatmap = ax.pcolormesh(X, Y, masked_perf, cmap=cmap, vmin=0, vmax=max_score, shading="nearest")
        cbar = fig.colorbar(heatmap, ax=ax, label="Mean Return")
        
        mask_color = np.ma.masked_array(np.zeros_like(grid_perf), mask=~mask_invalid)
        ax.pcolormesh(X, Y, mask_color, cmap=ListedColormap(["black"]), shading="nearest")

        contour_levels = np.linspace(0.1, 0.9, 5)
        if len(np.unique(grid_freq)) > 1:
            contours = ax.contour(X, Y, grid_freq, levels=contour_levels, colors="darkred", linewidths=2, alpha=0.9)
            ax.clabel(contours, inline=True, fontsize=10, fmt="%.1f")
        
        ax.set_title(f"CuReMix Condition Overlay at {int(pct*100)}% Training")
        ax.set_xlabel("RND Weight")
        ax.set_ylabel("ICM Weight")
        ax.set_xticks(grid_indices)
        ax.set_yticks(grid_indices)
        
        ax.grid(color="white", linestyle=":", alpha=0.3)
        plt.tight_layout()
        
        # Save frame to buffer for GIF
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        gif_frames.append(imageio.v2.imread(buf))
        
        # If this is a pdf target, save to file
        for target_pct in pdf_pcts:
            if np.isclose(pct, target_pct):
                filename = f"plot_4_contour_{int(pct*100)}pct_{achievement_filter}.pdf"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, format="pdf", bbox_inches="tight")
                print(f"Saved {filename} to {save_path}")
                break
                
        plt.close(fig)

    gif_filename = f"plot_4_contour_timelapse_{achievement_filter}.gif"
    gif_save_path = os.path.join(save_dir, gif_filename)
    imageio.mimsave(gif_save_path, gif_frames, fps=2, duration=500) # 2 fps
    print(f"Saved GIF to {gif_save_path}")
