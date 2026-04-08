"""Export minimal local artifacts needed to reproduce paper plots.

This script reads local Orbax checkpoints via ``load_local_orbax_data`` and writes
only the compact, plot-relevant information to ``artifacts/filtered_results``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from curemix.experiments.plots.create_plots_from_local import load_local_orbax_data


# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_ARTIFACTS_DIR = "artifacts/training_results"
OUTPUT_ARTIFACTS_DIR = "artifacts/filtered_results"
ACHIEVEMENT_FILTER = "defeat_skeleton+make_stone_pickaxe"


def _sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _export_single_achievement(df: pd.DataFrame, base_output: Path) -> None:
    """Write one achievement's compact run metadata and time series."""
    histories_dir = base_output / "histories"
    alpha_histories_dir = base_output / "alpha_histories"

    histories_dir.mkdir(parents=True, exist_ok=True)
    alpha_histories_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for _, row in df.iterrows():
        run_id = str(row["run_id"])
        safe_run_id = _sanitize_filename(run_id)

        history_file = histories_dir / f"{safe_run_id}.csv"
        history = row.get("history")
        if isinstance(history, pd.DataFrame) and not history.empty:
            history.to_csv(history_file, index=False)
        else:
            history_file = None

        alpha_history_file = None
        alpha_history = row.get("alpha_history")
        if isinstance(alpha_history, pd.DataFrame) and not alpha_history.empty:
            alpha_history_file = alpha_histories_dir / f"{safe_run_id}.csv"
            alpha_history.to_csv(alpha_history_file, index=False)

        metadata_rows.append(
            {
                "run_id": run_id,
                "run_type": row["run_type"],
                "icm_weight": float(row["icm_weight"]),
                "rnd_weight": float(row["rnd_weight"]),
                "extrinsic_weight": float(row["extrinsic_weight"]),
                "seed": int(row["seed"]),
                "final_performance": float(row["final_performance"]),
                "final_achievements": (
                    float(row["final_achievements"]) if pd.notna(row["final_achievements"]) else float("nan")
                ),
                "history_file": str(history_file.relative_to(base_output)) if history_file else "",
                "alpha_history_file": (str(alpha_history_file.relative_to(base_output)) if alpha_history_file else ""),
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = base_output / "runs_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"Exported {len(metadata_df)} runs to {base_output}")
    print(f"Metadata file: {metadata_path}")


def export_filtered_results(
    source_dir: str,
    output_dir: str,
    achievement_filter: str | None,
) -> list[Path]:
    """Export compact plot-relevant data for one or all achievements."""
    source_root = Path(source_dir)
    if not source_root.exists():
        raise ValueError(f"Source directory does not exist: {source_root}")

    candidate_achievements = [p.name for p in source_root.iterdir() if p.is_dir()]
    if not candidate_achievements:
        raise ValueError(f"No achievement directories found in {source_root}")

    achievements_to_export = []
    if achievement_filter and (source_root / achievement_filter).exists():
        achievements_to_export = [achievement_filter]
    else:
        achievements_to_export = candidate_achievements
        if achievement_filter:
            print(
                f"Requested achievement '{achievement_filter}' not found. "
                "Exporting all available achievements instead."
            )

    exported_paths = []
    for achievement in achievements_to_export:
        df = load_local_orbax_data(source_dir, achievement)
        if df.empty:
            print(f"No valid data found for achievement: {achievement}")
            continue

        safe_achievement = _sanitize_filename(achievement)
        base_output = Path(output_dir) / safe_achievement
        _export_single_achievement(df, base_output)
        exported_paths.append(base_output)

    if not exported_paths:
        raise ValueError("No valid local data found. Nothing to export.")

    return exported_paths


if __name__ == "__main__":
    export_filtered_results(
        source_dir=SOURCE_ARTIFACTS_DIR,
        output_dir=OUTPUT_ARTIFACTS_DIR,
        achievement_filter=ACHIEVEMENT_FILTER,
    )
