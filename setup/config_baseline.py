import sys
import copy
from math import sqrt
import pathlib
import json
import os

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

from sisyphus import *
import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

Path = tk.Path

# ------------------------------ Recipes --------------------------------------

from recipe.baseline import TrainBaselineJob, RunBaselineJob, ScoreBaselineJob
from recipe.data import MakeTrueTargets, CreateOutOfDomainTestData
from recipe.rag import CreateRagCheckpoint


def main():
    config_path = pathlib.Path(gs.BASELINE_ROOT) / 'baseline' / 'configs'
    data_root = pathlib.Path(gs.BASELINE_ROOT) / 'data'
    data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'

    config_ktd = json.load(open(config_path / 'detection' / 'params.json'))
    config_ks = json.load(open(config_path / 'selection' / 'params.json'))
    config_rg = json.load(open(config_path / 'generation' / 'params.json'))

    config_decoding = json.load(open(config_path / 'generation' / 'generation_params.json'))

    config_ks['dataset_args']['negative_sample_method'] = "all"

    # Always run with beam search
    config_decoding['no_sample'] = True
    config_decoding['num_beams'] = 10

    base_model_name_or_path = 'facebook/bart-large'
    config_ktd['model_name_or_path'] = base_model_name_or_path
    config_ks['model_name_or_path'] = base_model_name_or_path
    config_rg['model_name_or_path'] = base_model_name_or_path

    learning_rate = 6.25e-6
    config_ktd['learning_rate'] = learning_rate
    config_ks['learning_rate'] = learning_rate
    config_rg['learning_rate'] = learning_rate

    # Detection
    config_ktd['per_gpu_train_batch_size'] = 4
    train_ktd_model = TrainBaselineJob('ktd_baseline', config_ktd, extra_time=4)
    run_ktd_model = RunBaselineJob(train_ktd_model.checkpoint, gpus=1, additional_args="--eval_only")
    score_ktd_model_job = ScoreBaselineJob(run_ktd_model.preds, dataset='val')

    tk.register_output(f'ktd_baseline.json', run_ktd_model.preds)
    tk.register_output(f'ktd_baseline.score.json', score_ktd_model_job.scores)

    # Selection
    config_ks['per_gpu_train_batch_size'] = 1
    config_ks['max_candidates_per_forward_eval'] = 8
    config_ks['dataset_args']['history_max_tokens'] = 384
    config_ks['per_gpu_train_batch_random_sample'] = True
    train_ks_model = TrainBaselineJob('ks_baseline', config_ks, extra_time=4)
    run_ks_model = RunBaselineJob(train_ks_model.checkpoint, labels=run_ktd_model.preds, gpus=1, additional_args="--eval_only --eval_all_snippets", split=8)
    score_ks_model_job = ScoreBaselineJob(run_ks_model.preds)
    tk.register_output(f'ks_baseline.json', run_ks_model.preds)
    tk.register_output(f'ks_baseline.score.json', score_ks_model_job.scores)

    # Generation
    train_rg_model = TrainBaselineJob('rg_baseline', config_rg)
    run_rg_model = RunBaselineJob(train_rg_model.checkpoint, labels=run_ks_model.preds, config=config_decoding, additional_args=f"--generate --generation_params_file params.json")
    tk.register_output(f'rg_baseline.json', run_rg_model.preds)
    score_rg_model_job = ScoreBaselineJob(run_rg_model.preds)
    tk.register_output('rg_baseline.score.json', score_rg_model_job.scores)

    # Run on test data
    data_eval_args = {
        'data_path': data_eval_root,
        'eval_split': 'test'
    }

    run_ktd_model = RunBaselineJob(train_ktd_model.checkpoint, **data_eval_args, gpus=1, additional_args="--eval_only")
    tk.register_output(f'ktd_baseline.test.json', run_ktd_model.preds)
    
    run_ks_model = RunBaselineJob(train_ks_model.checkpoint, labels=run_ktd_model.preds, **data_eval_args, gpus=1, additional_args="--eval_only --eval_all_snippets", split=16, extra_time=2)
    tk.register_output(f'ks_baseline.test.json', run_ks_model.preds)
    
    run_rg_model = RunBaselineJob(train_rg_model.checkpoint, labels=run_ks_model.preds, **data_eval_args, config=config_decoding, additional_args=f"--generate --generation_params_file params.json")
    tk.register_output(f'rg_baseline.test.json', run_rg_model.preds)
