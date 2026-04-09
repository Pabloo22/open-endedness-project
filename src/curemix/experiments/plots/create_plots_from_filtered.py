"""Create paper plots from compact filtered artifacts only."""

from pathlib import Path

from curemix.experiments.plots.create_plots_from_local import load_filtered_results
from curemix.experiments.plots.plot_functions import (
    plot_1_combined_heatmaps,
    plot_2_combined_learning_curves,
    plot_3_combined_curriculum_adaptation,
    plot_3_combined_curriculum_adaptation_2x2,
    plot_4_combined_training_stages,
    plot_4_combined_training_stages_sliding_window,
)

# ==========================================
# CONFIGURATION
# ==========================================
FILTERED_RESULTS_DIR = "artifacts/filtered_results"
IMAGES_DIR = "images"
ACHIEVEMENT_FILTER = (
    "all"  # Use "all" to process all available, or specify a specific name like "defeat_skeleton+make_stone_pickaxe"
)
SLIDING_WINDOW_SIZE = 0.05  # 5% window for sliding-window Plot 4


def _resolve_achievements(filtered_dir: str, requested_achievement: str) -> list[str]:
    base = Path(filtered_dir)
    if not base.exists():
        return []

    available = sorted([entry.name for entry in base.iterdir() if entry.is_dir()])
    if not available:
        return []

    if requested_achievement and requested_achievement.lower() != "all":
        if requested_achievement in available:
            return [requested_achievement]
        else:
            print(f"Requested filtered achievement '{requested_achievement}' not found. Using all instead.")
            return available

    return available


if __name__ == "__main__":
    selected_achievements = _resolve_achievements(FILTERED_RESULTS_DIR, ACHIEVEMENT_FILTER)
    if not selected_achievements:
        print(f"No filtered results found under {FILTERED_RESULTS_DIR}")
        raise SystemExit(1)

    combined_plot_data = {}

    for selected_achievement in selected_achievements:
        df = load_filtered_results(FILTERED_RESULTS_DIR, selected_achievement)
        if df.empty:
            print(f"No valid filtered data found for achievement: {selected_achievement}. Skipping.")
            continue

        combined_plot_data[selected_achievement] = df

        achievement_images_dir = Path(IMAGES_DIR) / selected_achievement
        achievement_images_dir.mkdir(parents=True, exist_ok=True)

        message = f"\nLoaded {len(df)} filtered runs for '{selected_achievement}'."
        # print(f"{message} Creating plots in {achievement_images_dir}...")

        # One Plot 4-style 1x4 figure per achievement in images/
        # plot_4_combined_training_stages(df, IMAGES_DIR, achievement_filter=selected_achievement)

        # # One Plot 4-style 2x4 figure with sliding window alpha distribution per achievement in images/
        # plot_4_combined_training_stages_sliding_window(
        #     df,
        #     IMAGES_DIR,
        #     achievement_filter=selected_achievement,
        #     window_size=SLIDING_WINDOW_SIZE,
        # )

        # plot_1_heatmaps(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        # plot_2_learning_curves(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        # plot_2_b_learning_curves(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        # plot_3_curriculum_adaptation(
        #     df,
        #     str(achievement_images_dir),
        #     achievement_filter=selected_achievement,
        #     include_baseline_performance=True,
        # )
        # plt.close("all")

    if combined_plot_data:
        images_root = Path(IMAGES_DIR)
        images_root.mkdir(parents=True, exist_ok=True)
        plot_1_combined_heatmaps(combined_plot_data, str(images_root))
        plot_2_combined_learning_curves(combined_plot_data, str(images_root))
        # plot_3_combined_curriculum_adaptation(combined_plot_data, str(images_root))
        plot_3_combined_curriculum_adaptation_2x2(combined_plot_data, str(images_root))

    print("\nDone. Plots generated from filtered artifacts.")
