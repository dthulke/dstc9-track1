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
    config_ks['embedding_loss'] = 'nll'
    config_ks['per_gpu_train_batch_size'] = 1
    config_ks['max_candidates_per_forward_eval'] = 8
    config_ks['dataset_args']['history_max_tokens'] = 384
    config_ks['dataset_args']['selection_type'] = 'all'
    config_ks['per_gpu_train_batch_random_sample'] = True

    train_ks_model = TrainBaselineJob('ks_roberta_nll', config_ks, extra_time=4)

    # Retrieval Augmented Detection
    initial_rag_checkpoint = CreateRagCheckpoint(train_ks_model.checkpoint, 'facebook/bart-large').checkpoint
    
    config_ktd['model_name_or_path'] = initial_rag_checkpoint
    config_ktd['is_rag_model'] = True
    config_ktd['knowledge_encoder_checkpoint'] = train_ks_model.checkpoint
    config_ktd['per_gpu_train_batch_size'] = 1
    config_ktd['per_gpu_eval_batch_size'] = 4
    config_ktd['gradient_accumulation_steps'] = 16
    # Only for the knowledge retrieval
    config_ktd['dataset_args']['history_max_tokens'] = 384
    config_ktd['embedding_loss'] = 'nll'

    train_ktd_model = TrainBaselineJob('ktd_rag_roberta_nll', config_ktd, gpus=2, distributed=False, extra_time=24)
    run_ktd_model = RunBaselineJob(train_ktd_model.checkpoint, additional_args=f"--eval_only")
    tk.register_output(f'ktd_rag_roberta_nll.json', run_ktd_model.preds)
    score_ktd_model_job = ScoreBaselineJob(run_ktd_model.preds)
    tk.register_output('ktd_rag_roberta_nll.score.json', score_ktd_model_job.scores)
