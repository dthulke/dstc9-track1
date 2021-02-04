import copy
import pathlib
import json
import os
import sys

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

from sisyphus import *
import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

Path = tk.Path

# ------------------------------ Recipes --------------------------------------

from recipe.baseline import Timer


def main():
    data_root = pathlib.Path(gs.BASELINE_ROOT) / 'data'
    data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'

    val_args = {
        "data_path": data_root,
        "dataset": "val"
    }

    test_args = {
        "data_path": data_eval_root,
        "dataset": "test"
    }

    
    for eval_args in [val_args, test_args]:
        dataset = eval_args["dataset"]

        trained_single_task_model = "/link/to/checkpoint"
        timer_selection = Timer(trained_single_task_model, **eval_args, num_samples=1000)
        tk.register_output(f"times_single_task_model_{dataset}.txt", timer_selection.times)

        trained_multi_task_model = "/link/to/checkpoint"
        timer_detection = Timer(
            trained_multi_task_model, 
            num_samples=1000, 
            **eval_args, 
            additional_args="--timer_task \"detection\""
        )
        tk.register_output(f"times_multi_task_model{dataset}.txt", timer_detection.times)
