import sys
import copy
from math import sqrt
import pathlib
import json

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

from sisyphus import *
import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

Path = tk.Path

# ------------------------------ Recipes --------------------------------------

from recipe.baseline import TrainBaselineJob, RunBaselineJob, ScoreBaselineJob

async def async_main():
    config_path = pathlib.Path(gs.BASELINE_ROOT) / 'baseline' / 'configs'

    config_ktd = json.load(open(config_path / 'detection' / 'params.json'))
    config_ks = json.load(open(config_path / 'selection' / 'params.json'))
    config_rg = json.load(open(config_path / 'generation' / 'params.json'))

    config_ktd['model_name_or_path'] = 'gpt2'
    config_ks['model_name_or_path'] = 'gpt2'
    config_rg['model_name_or_path'] = 'gpt2'

    # Set negative sampling to all (to avoid --negative_sample_method "all")
    config_ks['dataset_args']['negative_sample_method'] = "all"

    train_ktd = TrainBaselineJob('ktd', config_ktd)
    train_ks = TrainBaselineJob('ks', config_ks)
    train_rg = TrainBaselineJob('rg-hml128-kml128', config_rg)

    run_ktd = RunBaselineJob(train_ktd.checkpoint, additional_args="--eval_only")
    run_ks = RunBaselineJob(train_ks.checkpoint, labels=run_ktd.preds, gpus=2, additional_args="--eval_only --eval_all_snippets")
    run_rg = RunBaselineJob(train_rg.checkpoint, labels=run_ks.preds, additional_args=f"--generate --generation_params_file {config_path / 'generation/generation_params.json'}")
    tk.register_output(f'baseline_val.json', run_rg.preds)

    score_job = ScoreBaselineJob(run_rg.preds)
    tk.register_output('baseline_val.score.json', score_job.scores)

async def py():
    await async_main()
