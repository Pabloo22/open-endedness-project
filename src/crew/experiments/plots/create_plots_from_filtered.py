"""Create paper plots from compact filtered artifacts only."""

from pathlib import Path

from crew.experiments.plots.create_plots_from_local import load_filtered_results
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
FILTERED_RESULTS_DIR = "artifacts/filtered_results"
IMAGES_DIR = "images"
ACHIEVEMENT_FILTER = "defeat_skeleton+make_stone_pickaxe"


def _resolve_achievement(filtered_dir: str, requested_achievement: str) -> str | None:
    base = Path(filtered_dir)
    if not base.exists():
        return None

    if requested_achievement and (base / requested_achievement).exists():
        return requested_achievement

    available = sorted([entry.name for entry in base.iterdir() if entry.is_dir()])
    if not available:
        return None

    if requested_achievement and requested_achievement != available[0]:
        print(
            f"Requested filtered achievement '{requested_achievement}' not found. " f"Using '{available[0]}' instead."
        )

    return available[0]


if __name__ == "__main__":
    selected_achievement = _resolve_achievement(FILTERED_RESULTS_DIR, ACHIEVEMENT_FILTER)
    if not selected_achievement:
        print(f"No filtered results found under {FILTERED_RESULTS_DIR}")
        raise SystemExit(1)

    df = load_filtered_results(FILTERED_RESULTS_DIR, selected_achievement)
    if df.empty:
        print(f"No valid filtered data found for achievement: {selected_achievement}")
        raise SystemExit(1)

    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} filtered runs for '{selected_achievement}'. Creating plots...")

    plot_1_heatmaps(df, IMAGES_DIR, achievement_filter=selected_achievement)
    plot_2_learning_curves(df, IMAGES_DIR, achievement_filter=selected_achievement)
    plot_3_curriculum_adaptation(
        df,
        IMAGES_DIR,
        achievement_filter=selected_achievement,
        include_baseline_performance=True,
    )
    plot_5_heatmaps(df, IMAGES_DIR, achievement_filter=selected_achievement)
    plot_6_learning_curves(df, IMAGES_DIR, achievement_filter=selected_achievement)

    print("Done. Plots generated from filtered artifacts.")
