"""Create paper plots from compact filtered artifacts only."""

from pathlib import Path
import matplotlib.pyplot as plt

from crew.experiments.plots.create_plots_from_local import load_filtered_results
from crew.experiments.plots.plot_functions import (
    plot_1_heatmaps,
    plot_2_learning_curves,
    plot_2_b_learning_curves,
    plot_3_curriculum_adaptation,
    plot_4_contour_overlay,
    plot_5_heatmaps,
    plot_6_learning_curves,
)

# ==========================================
# CONFIGURATION
# ==========================================
FILTERED_RESULTS_DIR = "artifacts/filtered_results"
IMAGES_DIR = "images"
ACHIEVEMENT_FILTER = (
    "all"  # Use "all" to process all available, or specify a specific name like "defeat_skeleton+make_stone_pickaxe"
)


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

    for selected_achievement in selected_achievements:
        df = load_filtered_results(FILTERED_RESULTS_DIR, selected_achievement)
        if df.empty:
            print(f"No valid filtered data found for achievement: {selected_achievement}. Skipping.")
            continue

        achievement_images_dir = Path(IMAGES_DIR) / selected_achievement
        achievement_images_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\nLoaded {len(df)} filtered runs for '{selected_achievement}'. Creating plots in {achievement_images_dir}..."
        )

        plot_1_heatmaps(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        plot_2_learning_curves(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        plot_2_b_learning_curves(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        plot_3_curriculum_adaptation(
            df,
            str(achievement_images_dir),
            achievement_filter=selected_achievement,
            include_baseline_performance=True,
        )
        plot_4_contour_overlay(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        plot_5_heatmaps(df, str(achievement_images_dir), achievement_filter=selected_achievement)
        plot_6_learning_curves(df, str(achievement_images_dir), achievement_filter=selected_achievement)

        plt.close("all")

    print("\nDone. Plots generated from filtered artifacts.")
