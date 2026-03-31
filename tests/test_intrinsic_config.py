import os
import sys

sys.path.append("src")

try:
    from crew.hyp_tuning.wandb_hp_search import build_base_tuning_config

    config = build_base_tuning_config(
        tuning_phase="intrinsic",
        intrinsic_modules=("rnd",),
        entity=None,
        group=None,
        train_seed=0,
        total_timesteps=1000,
    )
    print("selected_intrinsic_modules:", config.selected_intrinsic_modules)
    print("baseline_fixed_training_alpha:", config.baseline_fixed_training_alpha)
    print("evaluation_alphas:", config.evaluation_alphas)
except Exception as e:
    import traceback

    traceback.print_exc()
