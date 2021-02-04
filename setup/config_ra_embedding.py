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
    ground_truth_labels = data_root / 'val' / 'labels.json'    
    data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'
    eval_ground_truth = data_eval_root / 'test' / 'labels.json'

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
    config_ks['per_gpu_train_batch_size'] = 1
    config_ks['max_candidates_per_forward_eval'] = 8
    config_ks['dataset_args']['history_max_tokens'] = 384
    config_ks['dataset_args']['selection_type'] = 'all'
    config_ks['per_gpu_train_batch_random_sample'] = True

    losses = ["nll", "triplet"]

    for loss in losses:
        config_ks['embedding_loss'] = loss
    
        train_ks_model = TrainBaselineJob('ks_baseline', config_ks, gpus=2, extra_time=24)
        for dataset, ground_truth, data_root in zip(['val', 'test'], [ground_truth_labels, eval_ground_truth], [data_root, data_eval_root]):

            run_ks_model = RunBaselineJob(
                train_ks_model.checkpoint, 
                labels=ground_truth, 
                gpus=1, 
                data_path=data_root, 
                eval_split=dataset,
                additional_args="--embedding --eval_all_snippets",
                extra_time=4
            )

            score_ks_model_job = ScoreBaselineJob(run_ks_model.preds, data_path=data_root, dataset=dataset)
            tk.register_output(f'ks_roberta_{loss}_{dataset}.json', run_ks_model.preds)
            tk.register_output(f'ks_roberta_{loss}_{dataset}.score.json', score_ks_model_job.scores)

        # RAG Generation
        initial_rag_checkpoint = CreateRagCheckpoint(train_ks_model.checkpoint, 'facebook/bart-large').checkpoint
    
        config_rg['model_name_or_path'] = initial_rag_checkpoint
        config_rg['is_rag_model'] = True
        config_rg['force_correct_knowledge_in_training'] = True
        config_rg['knowledge_encoder_checkpoint'] = train_ks_model.checkpoint
        config_rg['per_gpu_train_batch_size'] = 1
        config_rg['per_gpu_eval_batch_size'] = 4
        config_rg['gradient_accumulation_steps'] = 16
        # Only for the knowledge retrieval
        config_rg['dataset_args']['history_max_tokens'] = 384
        config_rg['embedding_loss'] = loss

        train_rg_model = TrainBaselineJob(f'rg_rag_roberta_{loss}', config_rg, gpus=2, distributed=False)

        for dataset, ground_truth, data_root in zip(['val', 'test'], [ground_truth_labels, eval_ground_truth], [data_root, data_eval_root]):

            run_rg_model = RunBaselineJob(
                train_rg_model.checkpoint, 
                labels=ground_truth, 
                config=config_decoding, 
                additional_args=f"--generate --generation_params_file params.json"
            )
            tk.register_output(f'rg_rag_roberta_{loss}_{dataset}.json', run_rg_model.preds)
            score_rg_model_job = ScoreBaselineJob(run_ks_model.preds, data_path=data_root, dataset=dataset)
            tk.register_output(f'rg_rag_roberta_{loss}_{dataset}.score.json', score_rg_model_job.scores)
