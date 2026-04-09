import os
import io
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from adjustText import adjust_text
import matplotlib.patheffects as pe


# Opacity used for uncertainty/shaded regions in line plots.
SHADED_REGION_ALPHA = 0.10

# Z value for two-sided 95% confidence intervals.
CI_95_Z = 1.96

# Font size used for legend and direct labels
LABEL_FONTSIZE = 5

# Figure size used for line plots
LINE_PLOT_FIGSIZE = (2.73, 2.2)  # More space for labels
DOUBLE_COLUMN_FIGSIZE = (6.3, 2.5)  # Use for wider plots that span the page
GIF_FIGSIZE = (3.03 * 2, 2.2 * 2)  # Use for animated GIFs to keep file size smaller

plt.rcParams.update(
    {
        # --- Fonts ---
        # ACL requires Times Roman.
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
        # --- Font Sizes ---
        # Main text is 10pt, so figure text should be slightly smaller (8-9pt)
        "font.size": 9,
        "axes.labelsize": 9,  # Axis labels
        "axes.titlesize": 9,  # Title size
        "xtick.labelsize": 8,  # Tick labels (slightly smaller to save space)
        "ytick.labelsize": 8,
        "legend.fontsize": LABEL_FONTSIZE,
        # --- Sizing ---
        # Single column width in ACL is exactly 7.7 cm (3.03 inches)
        # Double column width (spanning the page) is exactly 16 cm (6.3 inches)
        "figure.figsize": (3.03, 2.2),  # Use (6.3, 2.5) for a double-column figure
        "figure.dpi": 300,
        # --- Spacing & Margins ---
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,  # ACL papers need tight spacing to save page limit
        # --- Lines & Ticks ---
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def _add_curve_to_axis(ax, x, y, lower, upper, label, color, use_direct_labels, texts_list):
    """Helper to plot a curve with an uncertainty band and optional direct labels."""
    ax.plot(x, y, label=label, color=color)
    if lower is not None and upper is not None:
        ax.fill_between(x, lower, upper, color=color, alpha=SHADED_REGION_ALPHA)

    if use_direct_labels and len(x) > 0:
        last_x = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        last_y = y.iloc[-1] if hasattr(y, "iloc") else y[-1]
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
            fontsize=LABEL_FONTSIZE,
        )
        if texts_list is not None:
            texts_list.append(t)
    return x.max() if len(x) > 0 else 0


def _get_curemix_stats(curr_search_df, metric_col):
    """Helper to compute best-per-step max returns across alpha subsets for curriculum runs."""
    if curr_search_df is None or curr_search_df.empty:
        return None

    all_curr_histories = []
    for _, row in curr_search_df.iterrows():
        hist = row.get("history")
        if hist is None or hist.empty or metric_col not in hist.columns:
            continue
        hist = hist.copy()
        hist["seed"] = row.get("seed", 0)
        all_curr_histories.append(hist)

    if not all_curr_histories:
        return None

    curr_histories_df = pd.concat(all_curr_histories)
    best_per_step_seed = curr_histories_df.groupby(["eval/total_steps", "seed"])[metric_col].max().reset_index()
    histories = best_per_step_seed[["eval/total_steps", metric_col]].copy()
    histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
    return histories.groupby("eval/total_steps")[metric_col].agg(["mean", "min", "max"]).reset_index()


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
    grid_fixed, mask_invalid_fixed, grid_indices = _build_plot_1_grid(df, metric_col, "baseline", grid_size)
    grid_curr, mask_invalid_curr, _ = _build_plot_1_grid(df, metric_col, "curriculum", grid_size)

    cmap = sns.color_palette("cividis", as_cmap=True)
    cmap.set_bad("white")

    plot_configs = [
        (grid_fixed, mask_invalid_fixed, f"{prefix}_heatmap_fixed_{achievement_filter}.pdf", False),
        (grid_curr, mask_invalid_curr, f"{prefix}_heatmap_curriculum_{achievement_filter}.pdf", True),
    ]

    apply_plot_1_customization = prefix == "plot_1"
    show_icm_label = (achievement_filter == "collect_iron") if apply_plot_1_customization else True
    show_colorbar = (achievement_filter == "place_furnace+make_iron_pickaxe") if apply_plot_1_customization else True

    for grid, mask, filename, is_curriculum in plot_configs:
        fig, ax = plt.subplots()
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
            cbar=show_colorbar,
            cbar_kws={"label": cbar_label} if show_colorbar else None,
            annot_kws={"fontsize": 6},
            ax=ax,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )

        sns.heatmap(
            mask,
            mask=~mask,
            cmap=ListedColormap(["black"]),
            cbar=False,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            ax=ax,
            linewidths=0,
            rasterized=True,
        )

        hide_curriculum_rnd_label = apply_plot_1_customization and is_curriculum
        ax.set_xlabel("" if hide_curriculum_rnd_label else "RND weight")
        ax.set_ylabel("ICM weight" if show_icm_label else "")
        ax.tick_params(left=False, bottom=False)
        ax.tick_params(axis="x", rotation=45)
        ax.invert_yaxis()

        plt.tight_layout()

        # Save the figure as a PDF
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")


def _build_plot_1_grid(df, metric_col, run_type, grid_size):
    """Build a Plot 1 heatmap grid and invalid-cell mask for a given run type."""
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.875 + step / 2, step), 3)
    grid_dim = len(grid_indices)

    grid = np.full((grid_dim, grid_dim), np.nan)
    mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)

    subset = df[df["run_type"] == run_type]
    if subset.empty or metric_col not in subset.columns:
        return grid, mask_invalid, grid_indices

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

    return grid, mask_invalid, grid_indices


def _format_achievement_name_for_title(achievement_name):
    return achievement_name.replace("_", " ").replace("+", " & ")


def plot_1_combined_heatmaps(achievement_dfs, save_dir, grid_size=8, filename="plot_1_combined_heatmaps.pdf"):
    """Create a single Plot 1 figure across achievements with shared labels and one colorbar."""
    if not achievement_dfs:
        print("Warning: No achievement data provided for combined Plot 1 figure.")
        return

    achievement_names = sorted(achievement_dfs.keys())
    num_achievements = len(achievement_names)
    run_types = [("baseline", "Fixed weights"), ("curriculum", "CuReMix (Ours)")]

    fig_width = max(6.8, 2.0 * num_achievements + 0.6)
    fig_height = 4.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=len(run_types),
        ncols=num_achievements + 1,
        width_ratios=[1] * num_achievements + [0.06],
        wspace=0.08,
        hspace=0.1,
    )

    axes = np.empty((len(run_types), num_achievements), dtype=object)
    shared_ax = None
    for row_idx in range(len(run_types)):
        for col_idx in range(num_achievements):
            if shared_ax is None:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                shared_ax = ax
            else:
                ax = fig.add_subplot(gs[row_idx, col_idx], sharex=shared_ax, sharey=shared_ax)
            axes[row_idx, col_idx] = ax

    cax = fig.add_subplot(gs[:, -1])

    cmap = sns.color_palette("cividis", as_cmap=True)
    cmap.set_bad("white")

    for col_idx, achievement in enumerate(achievement_names):
        df = achievement_dfs[achievement]
        max_score = len(achievement.split("+")) if achievement else 1.0
        max_score = max(max_score, 1.0)
        title_name = _format_achievement_name_for_title(achievement)

        for row_idx, (run_type, _) in enumerate(run_types):
            ax = axes[row_idx, col_idx]
            grid, mask_invalid, grid_indices = _build_plot_1_grid(df, "final_performance", run_type, grid_size)
            grid_pct = (grid / max_score) * 100.0

            ax.set_facecolor("lightgray")
            sns.heatmap(
                grid_pct,
                cmap=cmap,
                annot=True,
                fmt=".0f",
                vmin=0.0,
                vmax=100.0,
                xticklabels=grid_indices,
                yticklabels=grid_indices,
                cbar=False,
                annot_kws={"fontsize": 6},
                ax=ax,
                alpha=0.8,
                linewidths=0,
                rasterized=True,
            )

            sns.heatmap(
                mask_invalid,
                mask=~mask_invalid,
                cmap=ListedColormap(["black"]),
                cbar=False,
                xticklabels=grid_indices,
                yticklabels=grid_indices,
                ax=ax,
                linewidths=0,
                rasterized=True,
            )

            ax.tick_params(left=False, bottom=False)
            ax.tick_params(axis="x", rotation=45)
            if row_idx < len(run_types) - 1:
                ax.tick_params(labelbottom=False)
            if col_idx > 0:
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()

            ax.set_xlabel("")
            ax.set_ylabel("")

            if row_idx == 0:
                ax.set_title(title_name, fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=100.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Return (% of max)")

    fig.text(0.06, 0.69, run_types[0][1], rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")
    fig.text(0.06, 0.27, run_types[1][1], rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")

    fig.supxlabel("RND weight")
    fig.supylabel("ICM weight", x=0.02)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved combined Plot 1 heatmaps to {save_path}")
    plt.close(fig)


def plot_1_heatmaps(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    max_score = len(achievement_filter.split("+")) if achievement_filter else 1.0
    _plot_heatmaps_base(
        df,
        save_dir,
        metric_col="final_performance",
        prefix="plot_1",
        cbar_label="Mean final return",
        max_score=max_score,
        achievement_filter=achievement_filter,
        grid_size=grid_size,
    )


def _plot_learning_curves_base(df, save_dir, achievement_filter, metric_col, prefix, ylabel, use_direct_labels=True):
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

        if best_weights is not None:
            best_df = search_df[
                (search_df["icm_weight"] == best_weights[0]) & (search_df["rnd_weight"] == best_weights[1])
            ]
            return best_weights, best_df
        return None, None

    ext_only_df = df[(df["run_type"] == "baseline") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]

    fixed_search_df = df[(df["run_type"] == "baseline") & ~((df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0))]
    best_fixed_weights, best_fixed_df = get_best_run(fixed_search_df)

    # For curriculum, we pick the best alpha value for each timestep individually for each seed.
    curr_search_df = df[df["run_type"] == "curriculum"]
    best_curr_stats = _get_curemix_stats(curr_search_df, metric_col)

    def plot_curve(subset_df, label, color):
        if subset_df is None or subset_df.empty:
            return 0

        histories = pd.concat(subset_df["history"].tolist())
        histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
        stats = histories.groupby("eval/total_steps")[metric_col].agg(["mean", "min", "max"]).reset_index()

        return _add_curve_to_axis(
            ax,
            stats["eval/total_steps"],
            stats["mean"],
            stats["min"],
            stats["max"],
            label,
            color,
            use_direct_labels,
            texts_list,
        )

    max_steps = 0
    texts_list = []

    m = plot_curve(ext_only_df, "Extrinsic only", "#56B4E9")
    if m:
        max_steps = max(max_steps, m)

    if best_fixed_weights:
        m = plot_curve(best_fixed_df, f"Best fixed weights", "#E69F00")
        if m:
            max_steps = max(max_steps, m)

    if best_curr_stats is not None and not best_curr_stats.empty:
        m = _add_curve_to_axis(
            ax,
            best_curr_stats["eval/total_steps"],
            best_curr_stats["mean"],
            best_curr_stats["min"],
            best_curr_stats["max"],
            "CuReMix (Ours)",
            "#009E73",
            use_direct_labels,
            texts_list,
        )
        if m:
            max_steps = max(max_steps, m)

    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel(ylabel)
    if max_steps > 0:
        ax.set_xlim(0, max_steps)
    ax.grid(True, alpha=0.3)

    if texts_list and use_direct_labels:
        # Ensure accurate multiline text extents before adjustText computes overlaps.
        fig.canvas.draw()
        adjust_text(
            texts_list,
            ax=ax,
            only_move={"text": "y"},
            ensure_inside_axes=False,
            expand=(1.01, 1.2),
            force_text=(0.1, 10.0),
        )
    elif not use_direct_labels:
        loc = "best"
        ax.legend(loc=loc, frameon=False, fontsize=LABEL_FONTSIZE)

    # Save the figure as a PDF
    weights_str = f"ICM_{best_fixed_weights[0]}_RND_{best_fixed_weights[1]}" if best_fixed_weights else "no_fixed"
    save_path = os.path.join(
        save_dir,
        f"{prefix}_learning_curves_{achievement_filter}_{weights_str}.pdf",
    )
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved {prefix.replace('_', ' ').title()} to {save_path}")


def plot_2_learning_curves(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe"):
    _plot_learning_curves_base(
        df,
        save_dir,
        achievement_filter,
        metric_col="standardized_return_mean",
        prefix="plot_2",
        ylabel="Extrinsic Return (Mean)",
        use_direct_labels="collect_iron" in achievement_filter,
    )


def plot_2_b_learning_curves(
    df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", use_direct_labels=True
):
    fig, ax = plt.subplots(figsize=LINE_PLOT_FIGSIZE)

    metric_col = "standardized_return_mean"
    final_metric_col = "final_performance"
    prefix = "plot_2_b"
    ylabel = "Extrinsic return (mean)"

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
            "Top 0-20%",
            "Top 20-40%",
            "Top 40-60%",
            "Top 60-80%",
            "Top 80-100%",
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

                    last_step = _add_curve_to_axis(
                        ax,
                        stats["eval/total_steps"],
                        stats["mean"],
                        lower_bound,
                        upper_bound,
                        labels[q_idx],
                        colors[q_idx],
                        use_direct_labels,
                        texts_list,
                    )
                    max_steps = max(max_steps, last_step)
    # CuReMix
    curr_search_df = df[df["run_type"] == "curriculum"]
    best_curr_stats = _get_curemix_stats(curr_search_df, metric_col)

    if best_curr_stats is not None and not best_curr_stats.empty:
        last_step = _add_curve_to_axis(
            ax,
            best_curr_stats["eval/total_steps"],
            best_curr_stats["mean"],
            best_curr_stats["min"],
            best_curr_stats["max"],
            "CuReMix (Ours)",
            "#009E73",
            use_direct_labels,
            texts_list,
        )
        max_steps = max(max_steps, last_step)

    ax.set_xlabel("Total environment steps")
    ax.set_ylabel(ylabel)
    if max_steps > 0:
        ax.set_xlim(0, max_steps)
    ax.grid(True, alpha=0.3)

    if texts_list and use_direct_labels:
        # Force renderer update so multiline labels have correct bounding boxes.
        fig.canvas.draw()
        adjust_text(
            texts_list,
            ax=ax,
            only_move={"text": "y"},
            ensure_inside_axes=False,
            expand=(1.01, 1.2),
            force_text=(0.1, 10.0),
            time_lim=5,
        )
    elif not use_direct_labels:
        ax.legend(loc="best", frameon=False, fontsize=LABEL_FONTSIZE)

    save_path = os.path.join(save_dir, f"{prefix}_learning_curves_{achievement_filter}.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved {prefix.replace('_', ' ').title()} to {save_path}")


def plot_2_combined_learning_curves(
    achievement_dfs,
    save_dir,
    filename="plot_2_combined_learning_curves.pdf",
):
    """Create one 1xN figure with Plot 2 and Plot 2b curves overlaid per achievement."""
    if not achievement_dfs:
        print("Warning: No achievement data provided for combined Plot 2 figure.")
        return

    metric_col = "standardized_return_mean"
    final_metric_col = "final_performance"

    achievement_names = sorted(achievement_dfs.keys())
    num_achievements = len(achievement_names)

    def _percent_of_max(values, max_score):
        return (values / max_score) * 100.0

    def _best_fixed_subset(df):
        fixed_search_df = df[(df["run_type"] == "baseline") & ~((df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0))]
        if fixed_search_df.empty:
            return None

        mean_perfs = fixed_search_df.groupby(["icm_weight", "rnd_weight"])[final_metric_col].mean().reset_index()
        if mean_perfs.empty:
            return None

        max_perf = mean_perfs[final_metric_col].max()
        candidates = mean_perfs[mean_perfs[final_metric_col] >= max_perf - 0.02]

        best_metric = -np.inf
        best_weights = None
        for _, row in candidates.iterrows():
            icm, rnd = row["icm_weight"], row["rnd_weight"]
            cand_df = fixed_search_df[(fixed_search_df["icm_weight"] == icm) & (fixed_search_df["rnd_weight"] == rnd)]
            histories = pd.concat(cand_df["history"].tolist())
            if metric_col not in histories.columns:
                continue
            auc_surrogate = histories[metric_col].mean()
            if auc_surrogate > best_metric:
                best_metric = auc_surrogate
                best_weights = (icm, rnd)

        if best_weights is None:
            return None

        return fixed_search_df[
            (fixed_search_df["icm_weight"] == best_weights[0]) & (fixed_search_df["rnd_weight"] == best_weights[1])
        ]

    def _stats_from_subset(subset_df):
        if subset_df is None or subset_df.empty:
            return None
        histories = pd.concat(subset_df["history"].tolist())
        if metric_col not in histories.columns:
            return None
        histories["eval/total_steps"] = histories["eval/total_steps"].round(decimals=-4)
        return histories.groupby("eval/total_steps")[metric_col].agg(["mean", "min", "max"]).reset_index()

    prepared = {}

    quintile_colors = sns.color_palette("rocket", 5)
    quintile_labels = [
        "Top 0-20%",
        "Top 20-40%",
        "Top 40-60%",
        "Top 60-80%",
        "Top 80-100%",
    ]

    for achievement in achievement_names:
        df = achievement_dfs[achievement]
        max_score = max(float(len(achievement.split("+"))) if achievement else 1.0, 1.0)
        achievement_max_steps = 0

        top_curves = []
        bottom_curves = []

        # Top row: Plot 2 curves.
        ext_only_df = df[(df["run_type"] == "baseline") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]
        ext_stats = _stats_from_subset(ext_only_df)
        if ext_stats is not None and not ext_stats.empty:
            top_curves.append(
                {
                    "label": "Extrinsic only",
                    "color": "#56B4E9",
                    "x": ext_stats["eval/total_steps"],
                    "mean": _percent_of_max(ext_stats["mean"], max_score),
                    "lower": _percent_of_max(ext_stats["min"], max_score),
                    "upper": _percent_of_max(ext_stats["max"], max_score),
                }
            )
            achievement_max_steps = max(achievement_max_steps, float(ext_stats["eval/total_steps"].max()))

        best_fixed_df = _best_fixed_subset(df)
        best_fixed_stats = _stats_from_subset(best_fixed_df)
        if best_fixed_stats is not None and not best_fixed_stats.empty:
            top_curves.append(
                {
                    "label": "Best fixed weights",
                    "color": "#E69F00",
                    "x": best_fixed_stats["eval/total_steps"],
                    "mean": _percent_of_max(best_fixed_stats["mean"], max_score),
                    "lower": _percent_of_max(best_fixed_stats["min"], max_score),
                    "upper": _percent_of_max(best_fixed_stats["max"], max_score),
                }
            )
            achievement_max_steps = max(achievement_max_steps, float(best_fixed_stats["eval/total_steps"].max()))

        curr_search_df = df[df["run_type"] == "curriculum"]
        best_curr_stats = _get_curemix_stats(curr_search_df, metric_col)
        if best_curr_stats is not None and not best_curr_stats.empty:
            top_curves.append(
                {
                    "label": "CuReMix (Ours)",
                    "color": "#009E73",
                    "x": best_curr_stats["eval/total_steps"],
                    "mean": _percent_of_max(best_curr_stats["mean"], max_score),
                    "lower": _percent_of_max(best_curr_stats["min"], max_score),
                    "upper": _percent_of_max(best_curr_stats["max"], max_score),
                }
            )
            achievement_max_steps = max(achievement_max_steps, float(best_curr_stats["eval/total_steps"].max()))

        # Bottom row: Plot 2b curves.
        fixed_search_df = df[(df["run_type"] == "baseline") & ~((df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0))]
        if not fixed_search_df.empty:
            mean_perfs = fixed_search_df.groupby(["icm_weight", "rnd_weight"])[final_metric_col].mean().reset_index()
            mean_perfs = mean_perfs.sort_values(by=final_metric_col, ascending=False).reset_index(drop=True)

            indices_list = np.array_split(np.arange(len(mean_perfs)), 5)
            quintiles = [mean_perfs.iloc[indices] for indices in indices_list if len(indices) > 0]

            for q_idx, quintile_df in enumerate(quintiles):
                q_histories = []
                for _, row in quintile_df.iterrows():
                    icm, rnd = row["icm_weight"], row["rnd_weight"]
                    cand_df = fixed_search_df[
                        (fixed_search_df["icm_weight"] == icm) & (fixed_search_df["rnd_weight"] == rnd)
                    ]
                    q_histories.extend(cand_df["history"].tolist())

                if not q_histories:
                    continue

                histories = pd.concat(q_histories)
                if metric_col not in histories.columns:
                    continue
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

                bottom_curves.append(
                    {
                        "label": quintile_labels[q_idx],
                        "color": quintile_colors[q_idx],
                        "x": stats["eval/total_steps"],
                        "mean": _percent_of_max(stats["mean"], max_score),
                        "lower": _percent_of_max(lower_bound, max_score),
                        "upper": _percent_of_max(upper_bound, max_score),
                    }
                )
                achievement_max_steps = max(achievement_max_steps, float(stats["eval/total_steps"].max()))

        if best_curr_stats is not None and not best_curr_stats.empty:
            bottom_curves.append(
                {
                    "label": "CuReMix (Ours)",
                    "color": "#009E73",
                    "x": best_curr_stats["eval/total_steps"],
                    "mean": _percent_of_max(best_curr_stats["mean"], max_score),
                    "lower": _percent_of_max(best_curr_stats["min"], max_score),
                    "upper": _percent_of_max(best_curr_stats["max"], max_score),
                }
            )

        prepared[achievement] = {
            "top_curves": top_curves,
            "bottom_curves": bottom_curves,
            "max_steps": achievement_max_steps,
        }

    fig_width = max(6.8, 2.1 * num_achievements + 1.0)
    fig_height = 2.8
    fig, axes = plt.subplots(1, num_achievements, figsize=(fig_width, fig_height), sharey=True)

    if num_achievements == 1:
        axes = np.array([axes])

    for col_idx, achievement in enumerate(achievement_names):
        ax = axes[col_idx]
        data = prepared[achievement]

        for curve in data["top_curves"]:
            _add_curve_to_axis(
                ax,
                curve["x"],
                curve["mean"],
                curve["lower"],
                curve["upper"],
                curve["label"],
                curve["color"],
                False,
                None,
            )

        for curve in data["bottom_curves"]:
            _add_curve_to_axis(
                ax,
                curve["x"],
                curve["mean"],
                curve["lower"],
                curve["upper"],
                curve["label"],
                curve["color"],
                False,
                None,
            )

        title_name = _format_achievement_name_for_title(achievement)
        ax.set_title(title_name, fontsize=8)
        ax.grid(True, alpha=0.3)

        if col_idx > 0:
            ax.tick_params(labelleft=False)

    for col_idx, achievement in enumerate(achievement_names):
        max_steps = prepared[achievement]["max_steps"]
        if max_steps > 0:
            axes[col_idx].set_xlim(0, max_steps)

    for ax in axes:
        ax.set_ylim(0, 100)

    legend_handles = []
    legend_labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(handle)

    if legend_handles:
        legend_fontsize = max(LABEL_FONTSIZE + 2, 7)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.1,
            columnspacing=1.0,
        )

    fig.supxlabel("Total environment steps", y=0.13)
    fig.supylabel("Extrinsic return (% of max)", x=0.045)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.33, wspace=0.10)

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved combined Plot 2 figure to {save_path}")
    plt.close(fig)


def plot_3_curriculum_adaptation(
    df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", include_baseline_performance=False
):
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
    _add_curve_to_axis(
        ax1,
        steps,
        stats["alpha_ext"]["mean"].values,
        stats["alpha_ext"]["min"].values,
        stats["alpha_ext"]["max"].values,
        "Extrinsic",
        "#56B4E9",
        True,
        texts_list1,
    )

    # ICM
    _add_curve_to_axis(
        ax1,
        steps,
        stats["alpha_icm"]["mean"].values,
        stats["alpha_icm"]["min"].values,
        stats["alpha_icm"]["max"].values,
        "ICM",
        "#E69F00",
        True,
        texts_list1,
    )

    # RND
    _add_curve_to_axis(
        ax1,
        steps,
        stats["alpha_rnd"]["mean"].values,
        stats["alpha_rnd"]["min"].values,
        stats["alpha_rnd"]["max"].values,
        "RND",
        "#009E73",
        True,
        texts_list1,
    )

    ax1.set_xlabel("Total environment steps")
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

        last_x = _add_curve_to_axis(
            ax_perf,
            b_stats["eval/total_steps"],
            b_stats["mean"],
            b_stats["min"],
            b_stats["max"],
            "",
            "#CC79A7",
            False,
            None,
        )

        ax_perf.set_xlabel("Total environment steps")
        ax_perf.set_ylabel("Extrinsic return (mean)")
        ax_perf.set_ylim(bottom=0)
        ax_perf.set_xlim(0, last_x)
        ax_perf.grid(True, alpha=0.3)

        save_path_perf = os.path.join(save_dir, f"plot_3_curriculum_adaptation_performance_{achievement_filter}.pdf")
        fig2.savefig(save_path_perf, format="pdf", bbox_inches="tight")
        print(f"Saved Plot 3 Performance to {save_path_perf}")


def plot_3_combined_curriculum_adaptation(
    achievement_dfs,
    save_dir,
    filename="plot_3_combined_curriculum_adaptation.pdf",
):
    """Create a single 1x4 Plot 3 figure with shared axes and external dual legends."""
    return _plot_3_combined_curriculum_adaptation_layout(
        achievement_dfs,
        save_dir,
        filename=filename,
        layout="1x4",
    )


def plot_3_combined_curriculum_adaptation_2x2(
    achievement_dfs,
    save_dir,
    filename="plot_3_combined_curriculum_adaptation_2x2.pdf",
):
    """Create a single 2x2 Plot 3 figure with shared axes and external dual legends."""
    return _plot_3_combined_curriculum_adaptation_layout(
        achievement_dfs,
        save_dir,
        filename=filename,
        layout="2x2",
    )


def _plot_3_combined_curriculum_adaptation_layout(
    achievement_dfs,
    save_dir,
    filename,
    layout,
):
    """Internal helper to render combined Plot 3 in either 1x4 or 2x2 layout."""
    if not achievement_dfs:
        print("Warning: No achievement data provided for combined Plot 3 figure.")
        return

    achievement_names = sorted(achievement_dfs.keys())
    if layout == "1x4":
        nrows, ncols = 1, 4
        fig_width, fig_height = 11.2, 3.3
        left, right, top, bottom = 0.08, 0.94, 0.84, 0.20
        wspace, hspace = 0.13, 0.00
        left_label_x = 0.04
        right_label_x = 0.985
        legend_y = 0.965
        xlabel_y = 0.0
    elif layout == "2x2":
        nrows, ncols = 2, 2
        fig_width, fig_height = 6.2, 5.7
        left, right, top, bottom = 0.10, 0.90, 0.88, 0.15
        wspace, hspace = 0.20, 0.22
        left_label_x = 0.03
        right_label_x = 0.97
        legend_y = 0.97
        xlabel_y = 0.05
    else:
        raise ValueError(f"Unsupported plot 3 layout '{layout}'. Expected '1x4' or '2x2'.")

    max_panels = nrows * ncols

    if len(achievement_names) > max_panels:
        print(
            f"Warning: Received {len(achievement_names)} achievements for combined Plot 3; "
            f"only first {max_panels} will be shown."
        )
        achievement_names = achievement_names[:max_panels]

    def _get_combined_stats(df, max_score):
        curr_df = df[(df["run_type"] == "curriculum") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]
        if curr_df.empty:
            return None

        all_alpha_histories = []
        for _, row in curr_df.iterrows():
            alpha_hist = row.get("alpha_history")
            if alpha_hist is None or alpha_hist.empty:
                continue
            hist_copy = alpha_hist.copy()
            hist_copy["seed"] = row.get("seed", 0)
            all_alpha_histories.append(hist_copy)

        if not all_alpha_histories:
            return None

        combined_alpha = pd.concat(all_alpha_histories)
        if "run/total_env_steps" not in combined_alpha.columns:
            return None
        combined_alpha["run/total_env_steps"] = combined_alpha["run/total_env_steps"].round(decimals=-4)

        alpha_stats = (
            combined_alpha.groupby("run/total_env_steps")[["alpha_ext", "alpha_icm", "alpha_rnd"]]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        perf_histories = []
        for _, row in curr_df.iterrows():
            hist = row.get("history")
            if hist is None or hist.empty or "standardized_return_mean" not in hist.columns:
                continue
            perf_histories.append(hist)

        if not perf_histories:
            return None

        perf_df = pd.concat(perf_histories)
        if "eval/total_steps" not in perf_df.columns:
            return None
        perf_df["eval/total_steps"] = perf_df["eval/total_steps"].round(decimals=-4)
        perf_stats = (
            perf_df.groupby("eval/total_steps")["standardized_return_mean"].agg(["mean", "min", "max"]).reset_index()
        )

        perf_stats_pct = perf_stats.copy()
        perf_stats_pct[["mean", "min", "max"]] = (perf_stats_pct[["mean", "min", "max"]] / max_score) * 100.0

        max_steps = 0
        if not alpha_stats.empty:
            max_steps = max(max_steps, float(alpha_stats["run/total_env_steps"].max()))
        if not perf_stats_pct.empty:
            max_steps = max(max_steps, float(perf_stats_pct["eval/total_steps"].max()))

        return {
            "alpha_stats": alpha_stats,
            "perf_stats_pct": perf_stats_pct,
            "max_steps": max_steps,
        }

    prepared = {}
    global_max_steps = 0
    for achievement in achievement_names:
        df = achievement_dfs[achievement]
        max_score = max(float(len(achievement.split("+"))) if achievement else 1.0, 1.0)
        prepared[achievement] = _get_combined_stats(df, max_score)
        if prepared[achievement] is not None:
            global_max_steps = max(global_max_steps, prepared[achievement]["max_steps"])

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=False, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)

    legend_handles = []
    legend_labels = []

    weight_curves = [
        ("alpha_ext", "Extrinsic", "#56B4E9"),
        ("alpha_icm", "ICM", "#E69F00"),
        ("alpha_rnd", "RND", "#CC79A7"),
    ]
    perf_color = "#009E73"

    for panel_idx in range(max_panels):
        ax = axes[panel_idx]
        ax2 = ax.twinx()

        if panel_idx >= len(achievement_names):
            ax.set_visible(False)
            ax2.set_visible(False)
            continue

        achievement = achievement_names[panel_idx]
        title_name = _format_achievement_name_for_title(achievement)
        ax.set_title(title_name, fontsize=8)

        data = prepared[achievement]
        if data is None:
            ax.text(0.5, 0.5, "No curriculum data", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        else:
            alpha_stats = data["alpha_stats"]
            perf_stats_pct = data["perf_stats_pct"]

            for key, label, color in weight_curves:
                if key not in alpha_stats.columns.get_level_values(0):
                    continue
                _add_curve_to_axis(
                    ax,
                    alpha_stats["run/total_env_steps"],
                    alpha_stats[key]["mean"],
                    alpha_stats[key]["min"],
                    alpha_stats[key]["max"],
                    label,
                    color,
                    False,
                    None,
                )
                ax.lines[-1].set_linestyle("--")

            _add_curve_to_axis(
                ax2,
                perf_stats_pct["eval/total_steps"],
                perf_stats_pct["mean"],
                perf_stats_pct["min"],
                perf_stats_pct["max"],
                "Return (% of max)",
                perf_color,
                False,
                None,
            )

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax2.set_ylim(0, 100.0)

        col_idx = panel_idx % ncols
        if col_idx > 0:
            ax.tick_params(labelleft=False)
        if col_idx < ncols - 1:
            ax2.tick_params(labelright=False)

        if data is not None and data["max_steps"] > 0:
            ax.set_xlim(0, data["max_steps"])
        elif global_max_steps > 0:
            ax.set_xlim(0, global_max_steps)

        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(handle)

        handles2, labels2 = ax2.get_legend_handles_labels()
        for handle, label in zip(handles2, labels2):
            if label not in legend_labels:
                legend_labels.append(label)
                legend_handles.append(handle)

    fig.supxlabel("Total environment steps", y=xlabel_y)
    fig.supylabel("Reward weights", x=left_label_x)
    fig.text(right_label_x, 0.5, "Extrinsic return (% of max)", rotation=270, va="center", ha="center")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=max(LABEL_FONTSIZE + 3, 8),
            handlelength=2.1,
            columnspacing=1.0,
        )

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved combined Plot 3 figure to {save_path}")
    plt.close(fig)


def _plot_4_base(
    df,
    save_dir,
    achievement_filter="place_furnace+make_iron_pickaxe",
    grid_size=8,
    heatmap_type="perf",
    show_contours=True,
    prefix="plot_4_contour",
):

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
                    continue

                match = curr_eval_df[
                    (np.isclose(curr_eval_df["icm_weight"], icm, atol=1e-3))
                    & (np.isclose(curr_eval_df["rnd_weight"], rnd, atol=1e-3))
                ]
                if match.empty:
                    continue

                seed_perfs = []
                for _, row in match.iterrows():
                    hist = row.get("history")
                    if hist is None or hist.empty or metric_col not in hist.columns:
                        continue

                    valid_hists = hist[hist["eval/total_steps"] <= target_step]
                    if not valid_hists.empty:
                        seed_perfs.append(valid_hists.iloc[-1][metric_col])

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

        # fig, ax = plt.subplots(figsize=GIF_FIGSIZE)
        fig, ax = plt.subplots()
        ax.set_facecolor("lightgray")

        cmap = sns.color_palette("cividis", as_cmap=True)
        cmap.set_bad("white")

        data_to_plot = grid_perf if heatmap_type == "perf" else grid_freq
        vmax = max_score if heatmap_type == "perf" else 1.0
        cbar_label = "Mean return" if heatmap_type == "perf" else "Alpha frequency"

        sns.heatmap(
            data_to_plot,
            cmap=cmap,
            annot=False,
            fmt=".2f",
            vmin=0.0,
            vmax=vmax,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            cbar_kws={"label": cbar_label},
            annot_kws={"fontsize": 6},
            ax=ax,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )

        sns.heatmap(
            mask_invalid,
            mask=~mask_invalid,
            cmap=ListedColormap(["black"]),
            cbar=False,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            ax=ax,
            linewidths=0,
            rasterized=True,
        )

        if show_contours:
            X, Y = np.meshgrid(np.arange(0.5, grid_dim + 0.5, 1), np.arange(0.5, grid_dim + 0.5, 1))

            contour_levels = np.linspace(0.1, 0.9, 5)
            if len(np.unique(grid_freq)) > 1:
                black_line_width = 1.5
                ax.contour(
                    X, Y, grid_freq, levels=contour_levels, colors="black", linewidths=black_line_width, alpha=1.0
                )
                contours = ax.contour(X, Y, grid_freq, levels=contour_levels, colors="white", linewidths=1, alpha=1.0)
                # Add labels with text outline for better visibility
                labels = ax.clabel(contours, inline=True, fontsize=5, fmt="%.1f")
                plt.setp(labels, path_effects=[pe.withStroke(linewidth=black_line_width, foreground="black")])

                # Identify the path of the lowest contour (0.1)
                try:
                    allsegs = contours.allsegs[0]  # segments for first (lowest) level
                    if len(allsegs) > 0:
                        vertices = np.concatenate([seg for seg in allsegs])
                        # Find the point on the contour that sits closest to the top right
                        target_pt = vertices[np.argmax(vertices[:, 0] + vertices[:, 1])]

                        ax.annotate(
                            "Sampling frequency",
                            xy=(target_pt[0], target_pt[1]),
                            xytext=(grid_dim - 0.2, grid_dim - 0.5),  # Top right
                            xycoords="data",
                            textcoords="data",
                            ha="right",
                            va="center",
                            fontsize=7,
                            fontweight="bold",
                            color="white",
                            arrowprops=dict(arrowstyle="->", color="white", lw=1.5, shrinkA=0, shrinkB=3),
                        )
                except Exception as e:
                    print(f"Could not draw sampling frequency annotation arrow: {e}")

        ax.set_title(f"{int(pct*100)}% Training")
        ax.set_xlabel("RND weight")
        ax.set_ylabel("ICM weight")
        ax.tick_params(left=False, bottom=False)
        ax.tick_params(axis="x", rotation=45)
        ax.invert_yaxis()

        plt.tight_layout()

        # Save frame to buffer for GIF
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        gif_frames.append(imageio.v2.imread(buf))

        # If this is a pdf target, save to file
        for target_pct in pdf_pcts:
            if np.isclose(pct, target_pct):
                filename = f"{prefix}_{achievement_filter}.pdf"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, format="pdf", bbox_inches="tight")
                print(f"Saved {filename} to {save_path}")
                break

        plt.close(fig)

    gif_filename = f"{prefix}_timelapse_{achievement_filter}.gif"
    gif_save_path = os.path.join(save_dir, gif_filename)
    imageio.mimsave(gif_save_path, gif_frames, fps=2, duration=500) # 2 fps
    print(f"Saved GIF to {gif_save_path}")


def plot_4_combined_training_stages(
    df,
    save_dir,
    achievement_filter="place_furnace+make_iron_pickaxe",
    grid_size=8,
    percentages=(0.25, 0.50, 0.75, 1.00),
):
    """Create a single 1x4 Plot 4 figure showing training stages with shared labels and one colorbar."""
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.875 + step / 2, step), 3)
    grid_dim = len(grid_indices)

    max_score = max(float(len(achievement_filter.split("+"))) if achievement_filter else 1.0, 1.0)
    metric_col = "standardized_return_mean"

    curr_eval_df = df[df["run_type"] == "curriculum"]
    curr_df = df[(df["run_type"] == "curriculum") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]
    if curr_eval_df.empty:
        print("Warning: No curriculum data found for combined Plot 4 figure.")
        return

    max_eval_steps = 0
    for _, row in curr_eval_df.iterrows():
        hist = row.get("history")
        if hist is not None and not hist.empty and "eval/total_steps" in hist.columns:
            max_eval_steps = max(max_eval_steps, hist["eval/total_steps"].max())

    if max_eval_steps == 0:
        print("Warning: No evaluation steps found for combined Plot 4 figure.")
        return

    fig_width = 9.8
    fig_height = 2.8
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=5,
        width_ratios=[1, 1, 1, 1, 0.06],
        wspace=0.06,
    )

    axes = np.empty(4, dtype=object)
    shared_ax = None
    for idx in range(4):
        if shared_ax is None:
            ax = fig.add_subplot(gs[0, idx])
            shared_ax = ax
        else:
            ax = fig.add_subplot(gs[0, idx], sharex=shared_ax, sharey=shared_ax)
        axes[idx] = ax

    cax = fig.add_subplot(gs[0, -1])

    cmap = sns.color_palette("cividis", as_cmap=True)
    cmap.set_bad("white")

    for idx, pct in enumerate(percentages[:4]):
        ax = axes[idx]
        target_step = max_eval_steps * float(pct)

        grid_pct = np.full((grid_dim, grid_dim), np.nan)
        grid_freq = np.zeros((grid_dim, grid_dim))
        mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)

        for i, icm in enumerate(grid_indices):
            for j, rnd in enumerate(grid_indices):
                if round(icm + rnd, 3) > 0.9:
                    mask_invalid[i, j] = True
                    continue

                match = curr_eval_df[
                    (np.isclose(curr_eval_df["icm_weight"], icm, atol=1e-3))
                    & (np.isclose(curr_eval_df["rnd_weight"], rnd, atol=1e-3))
                ]
                if match.empty:
                    continue

                seed_perfs = []
                for _, row in match.iterrows():
                    hist = row.get("history")
                    if hist is None or hist.empty or metric_col not in hist.columns:
                        continue

                    valid_hists = hist[hist["eval/total_steps"] <= target_step]
                    if not valid_hists.empty:
                        seed_perfs.append(valid_hists.iloc[-1][metric_col])

                if seed_perfs:
                    mean_perf = np.mean(seed_perfs)
                    grid_pct[i, j] = (mean_perf / max_score) * 100.0

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

        ax.set_facecolor("lightgray")
        sns.heatmap(
            grid_pct,
            cmap=cmap,
            annot=False,
            fmt=".0f",
            vmin=0.0,
            vmax=100.0,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            cbar=False,
            ax=ax,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )

        sns.heatmap(
            mask_invalid,
            mask=~mask_invalid,
            cmap=ListedColormap(["black"]),
            cbar=False,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            ax=ax,
            linewidths=0,
            rasterized=True,
        )

        if len(np.unique(grid_freq)) > 1:
            x_coords, y_coords = np.meshgrid(np.arange(0.5, grid_dim + 0.5, 1), np.arange(0.5, grid_dim + 0.5, 1))
            contour_levels = np.linspace(0.1, 0.9, 5)
            black_line_width = 1.2

            ax.contour(
                x_coords,
                y_coords,
                grid_freq,
                levels=contour_levels,
                colors="black",
                linewidths=black_line_width,
                alpha=1.0,
            )
            contours = ax.contour(
                x_coords,
                y_coords,
                grid_freq,
                levels=contour_levels,
                colors="white",
                linewidths=0.8,
                alpha=1.0,
            )

            # Show contour values directly on the lines, as in the original Plot 4.
            labels = ax.clabel(contours, inline=True, fontsize=6, fmt="%.1f")
            plt.setp(labels, path_effects=[pe.withStroke(linewidth=black_line_width, foreground="black")])

            # Keep a direct label anchored near the top-right contour area.
            try:
                allsegs = contours.allsegs[0]
                if len(allsegs) > 0:
                    vertices = np.concatenate([seg for seg in allsegs])
                    target_pt = vertices[np.argmax(vertices[:, 0] + vertices[:, 1])]
                    ax.annotate(
                        "Sampling frequency",
                        xy=(target_pt[0], target_pt[1]),
                        xytext=(grid_dim - 0.25, grid_dim - 0.55),
                        xycoords="data",
                        textcoords="data",
                        ha="right",
                        va="center",
                        fontsize=6,
                        fontweight="bold",
                        color="white",
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.2, shrinkA=0, shrinkB=3),
                    )
            except (IndexError, ValueError):
                pass

        ax.set_title(f"{int(round(float(pct) * 100))}% of training", fontsize=8)
        ax.tick_params(left=False, bottom=False)
        ax.tick_params(axis="x", rotation=45)
        if idx > 0:
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=100.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Return (% of max)")

    fig.supxlabel("RND weight")
    fig.supylabel("ICM weight", x=0.02)
    fig.suptitle(_format_achievement_name_for_title(achievement_filter), y=0.98, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.20)

    filename = f"plot_4_combined_{achievement_filter}.pdf"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved combined Plot 4 figure to {save_path}")
    plt.close(fig)


def plot_4_combined_training_stages_sliding_window(
    df,
    save_dir,
    achievement_filter="place_furnace+make_iron_pickaxe",
    grid_size=8,
    percentages=(0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00),
    window_size=0.05,
):
    """Create an 8-plot (2x4) figure showing training stages with sliding window alpha distribution.

    For each percentage X, shows the distribution of alphas from (X-window_size)% to X% of training.
    """
    step = 1.0 / grid_size
    grid_indices = np.round(np.arange(0, 0.875 + step / 2, step), 3)
    grid_dim = len(grid_indices)

    max_score = max(float(len(achievement_filter.split("+"))) if achievement_filter else 1.0, 1.0)
    metric_col = "standardized_return_mean"

    curr_eval_df = df[df["run_type"] == "curriculum"]
    curr_df = df[(df["run_type"] == "curriculum") & (df["icm_weight"] == 0.0) & (df["rnd_weight"] == 0.0)]

    if curr_eval_df.empty:
        print("Warning: No curriculum data found for combined Plot 4 sliding window figure.")
        return

    max_eval_steps = 0
    for _, row in curr_eval_df.iterrows():
        hist = row.get("history")
        if hist is not None and not hist.empty and "eval/total_steps" in hist.columns:
            max_eval_steps = max(max_eval_steps, hist["eval/total_steps"].max())

    if max_eval_steps == 0:
        print("Warning: No evaluation steps found for combined Plot 4 sliding window figure.")
        return

    fig_width = 11.2
    fig_height = 5.6
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        width_ratios=[1, 1, 1, 1, 0.06],
        wspace=0.03,
        hspace=0.12,
    )

    axes = np.empty(8, dtype=object)
    shared_ax = None
    for idx in range(8):
        row = idx // 4
        col = idx % 4
        if shared_ax is None:
            ax = fig.add_subplot(gs[row, col])
            shared_ax = ax
        else:
            ax = fig.add_subplot(gs[row, col], sharex=shared_ax, sharey=shared_ax)
        axes[idx] = ax

    cax = fig.add_subplot(gs[:, -1])

    cmap = sns.color_palette("cividis", as_cmap=True)
    cmap.set_bad("white")

    for idx, pct in enumerate(percentages[:8]):
        ax = axes[idx]
        target_step = max_eval_steps * float(pct)
        window_start_step = max(0, max_eval_steps * (float(pct) - window_size))

        grid_pct = np.full((grid_dim, grid_dim), np.nan)
        grid_freq = np.zeros((grid_dim, grid_dim))
        mask_invalid = np.zeros((grid_dim, grid_dim), dtype=bool)

        # Grid performance calculation (cumulative up to target_step)
        for i, icm in enumerate(grid_indices):
            for j, rnd in enumerate(grid_indices):
                if round(icm + rnd, 3) > 0.9:
                    mask_invalid[i, j] = True
                    continue

                match = curr_eval_df[
                    (np.isclose(curr_eval_df["icm_weight"], icm, atol=1e-3))
                    & (np.isclose(curr_eval_df["rnd_weight"], rnd, atol=1e-3))
                ]
                if match.empty:
                    continue

                seed_perfs = []
                for _, row in match.iterrows():
                    hist = row.get("history")
                    if hist is None or hist.empty or metric_col not in hist.columns:
                        continue

                    valid_hists = hist[hist["eval/total_steps"] <= target_step]
                    if not valid_hists.empty:
                        seed_perfs.append(valid_hists.iloc[-1][metric_col])

                if seed_perfs:
                    mean_perf = np.mean(seed_perfs)
                    grid_pct[i, j] = (mean_perf / max_score) * 100.0

        # Alpha frequency calculation using sliding window
        all_alpha_histories = []
        for _, row in curr_df.iterrows():
            alpha_hist = row.get("alpha_history")
            if alpha_hist is not None and not alpha_hist.empty and "run/total_env_steps" in alpha_hist.columns:
                # Use sliding window: from (target_step - window_size) to target_step
                valid_alpha = alpha_hist[
                    (alpha_hist["run/total_env_steps"] > window_start_step)
                    & (alpha_hist["run/total_env_steps"] <= target_step)
                ]
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

        ax.set_facecolor("lightgray")
        sns.heatmap(
            grid_pct,
            cmap=cmap,
            annot=False,
            fmt=".0f",
            vmin=0.0,
            vmax=100.0,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            cbar=False,
            ax=ax,
            alpha=0.8,
            square=True,
            linewidths=0,
            rasterized=True,
        )

        sns.heatmap(
            mask_invalid,
            mask=~mask_invalid,
            cmap=ListedColormap(["black"]),
            cbar=False,
            xticklabels=grid_indices,
            yticklabels=grid_indices,
            ax=ax,
            square=True,
            linewidths=0,
            rasterized=True,
        )

        if len(np.unique(grid_freq)) > 1:
            x_coords, y_coords = np.meshgrid(np.arange(0.5, grid_dim + 0.5, 1), np.arange(0.5, grid_dim + 0.5, 1))
            contour_levels = np.linspace(0.1, 0.9, 5)
            black_line_width = 1.2

            ax.contour(
                x_coords,
                y_coords,
                grid_freq,
                levels=contour_levels,
                colors="black",
                linewidths=black_line_width,
                alpha=1.0,
            )
            contours = ax.contour(
                x_coords,
                y_coords,
                grid_freq,
                levels=contour_levels,
                colors="white",
                linewidths=0.8,
                alpha=1.0,
            )

            # Show contour values directly on the lines
            labels = ax.clabel(contours, inline=True, fontsize=5, fmt="%.1f")
            plt.setp(labels, path_effects=[pe.withStroke(linewidth=black_line_width, foreground="black")])

            # Keep a direct label anchored near the top-right contour area
            try:
                allsegs = contours.allsegs[0]
                if len(allsegs) > 0:
                    vertices = np.concatenate([seg for seg in allsegs])
                    target_pt = vertices[np.argmax(vertices[:, 0] + vertices[:, 1])]
                    ax.annotate(
                        "Sampling frequency",
                        xy=(target_pt[0], target_pt[1]),
                        xytext=(grid_dim - 0.25, grid_dim - 0.55),
                        xycoords="data",
                        textcoords="data",
                        ha="right",
                        va="center",
                        fontsize=6,
                        fontweight="bold",
                        color="white",
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.8, shrinkA=0, shrinkB=3),
                    )
            except (IndexError, ValueError):
                pass

        # Title showing the window range
        window_start_pct = max(0, float(pct) - window_size) * 100
        window_end_pct = float(pct) * 100
        window_start_label = f"{window_start_pct:.1f}".rstrip("0").rstrip(".")
        window_end_label = f"{window_end_pct:.1f}".rstrip("0").rstrip(".")
        ax.set_title(f"From {window_start_label}% to {window_end_label}% of training", fontsize=7)
        ax.tick_params(left=False, bottom=False)
        ax.tick_params(axis="x", rotation=45)
        if idx < 4:
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
        else:
            ax.tick_params(axis="x", labelbottom=True)
        if idx % 4 > 0:
            ax.tick_params(labelleft=False)
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_ylabel("")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=100.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Return (% of max)")

    fig.supxlabel("RND weight", y=0.05)
    fig.supylabel("ICM weight", x=0.02)
    fig.suptitle(
        f"{_format_achievement_name_for_title(achievement_filter)}",
        y=0.98,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.15, wspace=0.03, hspace=0.12)

    filename = f"plot_4_combined_sliding_window_{achievement_filter}.pdf"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Saved combined Plot 4 sliding window figure to {save_path}")
    plt.close(fig)


def plot_4_contour_overlay(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    _plot_4_base(
        df, save_dir, achievement_filter, grid_size, heatmap_type="perf", show_contours=True, prefix="plot_4_contour"
    )


def plot_4_b(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    _plot_4_base(
        df, save_dir, achievement_filter, grid_size, heatmap_type="perf", show_contours=False, prefix="plot_4_b"
    )


def plot_4_c(df, save_dir, achievement_filter="place_furnace+make_iron_pickaxe", grid_size=8):
    _plot_4_base(
        df, save_dir, achievement_filter, grid_size, heatmap_type="freq", show_contours=False, prefix="plot_4_c"
    )
