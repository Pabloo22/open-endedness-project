import wandb
import re
import os
import pandas as pd
from crew.experiments.plots.plot_functions import plot_1_heatmaps, plot_2_learning_curves


# ==========================================
# CONFIGURATION
# ==========================================
WANDB_PROJECT_PATH = "openendedness-2026/open_end_proj"
ACHIEVEMENT_FILTER = "place_furnace+make_iron_pickaxe"
SAVE_DIR = "images"  # Directory for saving plots


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def parse_run_name(run_name):
    pattern = r"^(.*?)/(baseline|baselines|curriculum)/icm(.*)\+rnd(.*?)\|seed(\d+)$"
    match = re.match(pattern, run_name)
    if not match:
        return None

    achievement, run_type, icm_str, rnd_str, seed_str = match.groups()

    icm_weight = float(icm_str.replace("p", ".")) if icm_str else 0.0
    rnd_weight = float(rnd_str.replace("p", ".")) if rnd_str else 0.0
    seed = int(seed_str)

    # Normalize run_type string
    run_type = "baseline" if "baseline" in run_type else "curriculum"

    return achievement, run_type, icm_weight, rnd_weight, seed


def fetch_all_data(api):
    runs = api.runs(WANDB_PROJECT_PATH)
    data = []

    print(f"Fetching runs from {WANDB_PROJECT_PATH}...")
    for run in runs:
        parsed = parse_run_name(run.name)
        if not parsed or parsed[0] != ACHIEVEMENT_FILTER:
            continue

        achievement, run_type, icm, rnd, seed = parsed

        step_key = "eval/total_steps"
        return_key = "eval/return_mean" if run_type == "baseline" else "eval/(1.0, 0.0, 0.0)/return_mean"

        history = run.history(keys=[step_key, return_key], pandas=True)
        history = history.dropna(subset=[step_key, return_key])

        if history.empty:
            continue

        history = history.rename(columns={return_key: "standardized_return_mean"})
        final_performance = history["standardized_return_mean"].iloc[-1]

        data.append(
            {
                "run_id": run.id,
                "run_type": run_type,
                "icm_weight": icm,
                "rnd_weight": rnd,
                "extrinsic_weight": round(1.0 - icm - rnd, 2),
                "seed": seed,
                "final_performance": final_performance,
                "history": history,
            }
        )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Create the directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    api = wandb.Api()
    df = fetch_all_data(api)

    if not df.empty:
        print("Data fetched successfully. Generating Plot 1...")
        plot_1_heatmaps(df, SAVE_DIR, achievement_filter=ACHIEVEMENT_FILTER)

        print("Generating Plot 2...")
        plot_2_learning_curves(df, SAVE_DIR, achievement_filter=ACHIEVEMENT_FILTER)
    else:
        print("No data found matching the current filters.")
