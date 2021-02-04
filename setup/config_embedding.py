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
    test_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'
    ground_truth_labels = data_root / 'val' / 'labels.json'
    ground_truth_labels_test = test_root / 'test' / 'labels.json'

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

    # Selection
    config_ks['task'] = "embedding"
    config_ks['model_name_or_path'] = 'roberta-large'
    config_ks['embedding_loss'] = 'triplet'
    config_ks['per_gpu_train_batch_size'] = 1
    config_ks['max_candidates_per_forward_eval'] = 8
    config_ks['dataset_args']['history_max_tokens'] = 384
    config_ks['dataset_args']['selection_type'] = 'all'
    config_ks['per_gpu_train_batch_random_sample'] = True
    config_ks['num_train_epochs'] = 10

    train_ks_model = TrainBaselineJob('ks_roberta_large_triplet_hard', config_ks, extra_time=20)
    run_ks_model = RunBaselineJob(train_ks_model.checkpoint, labels=ground_truth_labels, gpus=1, additional_args="--embedding --eval_all_snippets", split = 8, extra_time=4)
    score_ks_model_job = ScoreBaselineJob(run_ks_model.preds)
    tk.register_output(f'ks_roberta_large_triplet_hard.json', run_ks_model.preds)
    tk.register_output(f'ks_roberta_large_triplet_hard.score.json', score_ks_model_job.scores)

    run_ks_model = RunBaselineJob(train_ks_model.checkpoint, labels=ground_truth_labels_test, data_path=test_root, eval_split="test", gpus=1, additional_args="--embedding --eval_all_snippets", split = 8, extra_time=4)
    score_ks_model_job = ScoreBaselineJob(run_ks_model.preds, data_path=test_root, dataset="test")
    tk.register_output(f'ks_roberta_large_triplet_hard.json', run_ks_model.preds)
    tk.register_output(f'ks_roberta_large_triplet_hard.score.json', score_ks_model_job.scores)
