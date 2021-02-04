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

def init_configs():
    config_path = pathlib.Path(gs.BASELINE_ROOT) / 'baseline' / 'configs'
    config = json.load(open(config_path / 'multitask' / 'params.json'))
    config_selection = json.load(open(config_path / 'joint_selection' / 'params.json'))

    config_selection['model_name_or_path'] = 'facebook/bart-large'
    config_selection['learning_rate'] = 6.25e-6
    config_selection['num_train_epochs'] = 10
    config_selection['dataset_args']['negative_sample_method'] = "all"
    config_selection['dataset_args']['history_max_tokens'] = 384
    config_selection['per_gpu_train_batch_size'] = 1
    config_selection['per_gpu_eval_batch_size'] = 1
    config_selection['per_gpu_train_batch_random_sample'] = True

    config_decoding = json.load(open(config_path / 'generation' / 'generation_params.json'))

    config['dataset_args']['negative_sample_method'] = "all"

    # Always run with beam search
    config_decoding['no_sample'] = True
    config_decoding['num_beams'] = 10

    base_model_name_or_path = 'facebook/bart-base'
    config['model_name_or_path'] = base_model_name_or_path

    learning_rate = 6.25e-6
    config['learning_rate'] = learning_rate
    config['dataset_args']['negative_sample_method'] = "all"
    config['dataset_args']['selection_type'] = "all"
    config['dataset_args']['history_max_tokens'] = 384
    config['per_gpu_train_batch_size'] = 1
    config['per_gpu_eval_batch_size'] = 1
    config['per_gpu_train_batch_random_sample'] = True

    config["num_train_epochs"] = 10

    return config, config_decoding

def main():
    data_root = pathlib.Path(gs.BASELINE_ROOT) / 'data'
    data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'
    ground_truth_labels = data_root / "val" / "labels.json"
    ground_truth_labels_test = data_eval_root / "test" / "labels.json"

    config, config_selection, config_decoding = init_configs()

    # Multitask model trained on all three tasks
    train_multitask = TrainBaselineJob(
        'bart-multitask',
        config,
        gpus=2,
        extra_memory=5,
        extra_time=60,
        additional_args='--multitask --skip_per_epoch_eval'
    )

    # Selection model trained to classify domain, entity, doc and the full data
    train_selection = TrainBaselineJob(
        'bart-joint-selection-four-tasks',
        config_selection,
        gpus=2,
        extra_memory=4,
        extra_time=10,
        additional_args='--multitask --selection'
    )

    for dataset, path, ground_truth in zip(['val', 'test'], [data_root, data_eval_root], [ground_truth_labels, ground_truth_labels_test]):

        run_ktd_model = RunBaselineJob(
            train_multitask.checkpoint,
            gpus=1,
            data_path=path,
            eval_split=dataset,
            additional_args="--eval_only --multitask --eval_task \"detection\""
        ) # specifying the eval_task is necessary for the multitask model
        score_ktd_model_job = ScoreBaselineJob(
            run_ktd_model.preds,
            dataset=dataset,
            data_path=path
        )

        tk.register_output(f'ktd_multitask_{dataset}.json', run_ktd_model.preds)
        tk.register_output(f'ktd_multitask_{dataset}.score.json', score_ktd_model_job.scores)

        # Selection model trained on multiple relevance classification tasks
        run_joint_selection = RunBaselineJob(
            train_selection.checkpoint,
            labels=ground_truth,
            data_path=data_root,
            eval_split="test",
            gpus=1,
            split=8,
            additional_args="--eval_only --eval_all_snippets --multitask --selection \
                             --eval_task \"selection\" --selection_type \"all\"")
        tk.register_output(f'joint_selection_{dataset}.json', run_joint_selection.preds)

        score_job_joint_selection = ScoreBaselineJob(
            run_joint_selection.preds,
            data_path=path,
            dataset=dataset
        )
        tk.register_output('ks_join_selection_{dataset}.score.json', score_job_joint_selection.scores)

        # Selection based on Multitask model trained on all three tasks
        run_ks_model = RunBaselineJob(
            train_multitask.checkpoint, 
            labels=ground_truth, 
            gpus=1, 
            extra_time=10, 
            data_path=path, 
            eval_split=dataset, 
            additional_args="--multitask --eval_task \"selection\" --eval_only --eval_all_snippets", 
            split=16
        )

        score_ks_model_job = ScoreBaselineJob(
            run_ks_model.preds, 
            data_path=path, 
            dataset=dataset
        )
        tk.register_output(f'ks_multitask.json', run_ks_model.preds)
        tk.register_output(f'ks_multitask.score.json', score_ks_model_job.scores)

        # Generation
        run_rg_model = RunBaselineJob(
            train_multitask.checkpoint, 
            labels=ground_truth, 
            config=config_decoding, 
            data_path=path, 
            eval_split=dataset, 
            additional_args=f"--generate --generation_params_file params.json --multitask --eval_task \"generation\""
        )

        tk.register_output(f'rg_multitask_{dataset}.json', run_rg_model.preds)
        score_rg_model_job = ScoreBaselineJob(
            run_rg_model.preds, 
            data_path=path, 
            dataset=dataset
        )
        tk.register_output('rg_multitask_{dataset}.score.json', score_rg_model_job.scores)
