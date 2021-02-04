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

from recipe.baseline import TrainBaselineJob, RerunSelectionSplit, RunBaselineJob, ScoreBaselineJob
from recipe.data import MakeTrueTargets, CreateOutOfDomainTestData
from recipe.rag import CreateRagCheckpoint

def run_stages(ks_model, selection_types, additional_args, **ground_truth_args):
    """Runs all selection stages for a selection model trained on all relevance classification tasks."""
    stage = RunBaselineJob(
        ks_model, 
        **ground_truth_args, 
        additional_args=additional_args + (f" --selection_type \"{selection_types[0]}\""), gpus=1, split=12
    )
    
    if "labels" in ground_truth_args:
        del ground_truth_args["labels"]

    for selection_type in selection_types[1:]:
        stage = RunBaselineJob(
            ks_model, 
            **ground_truth_args, 
            labels=stage.preds, 
            additional_args=additional_args + (f" --selection_type \"{selection_type}\""), 
            gpus=1, 
            split=12
        )

    return stage.preds

def run_stages_single_task(models, additional_args, **ground_truth_args):
    """Runs all selection stages for a set of models, each trained separately on one relevance classification task."""
    stage = RunBaselineJob(models[0].checkpoint, **ground_truth_args, additional_args=additional_args, gpus=1, split=6)

    for model in models[1:]:
        stage = RunBaselineJob(model.checkpoint, labels=stage.preds, additional_args=additional_args, gpus=1, split=6)

    return stage.preds

def main():
    config_path = pathlib.Path(gs.BASELINE_ROOT) / 'baseline' / 'configs'
    data_root = pathlib.Path(gs.BASELINE_ROOT) / 'data'
    ground_truth_labels = data_root / "val" / "labels.json"

    data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'
    eval_ground_truth = data_eval_root / "test" / "labels.json"

    config_ks = json.load(open(config_path / 'selection' / 'params.json'))
    config_ks['dataset_args']['negative_sample_method'] = "all"

    base_model_name_or_path = 'facebook/bart-large'
    config_ks['model_name_or_path'] = base_model_name_or_path
    learning_rate = 6.25e-6
    config_ks['learning_rate'] = learning_rate

    # Selection
    config_ks['per_gpu_train_batch_size'] = 1
    config_ks['per_gpu_eval_batch_size'] = 1
    config_ks['max_candidates_per_forward_eval'] = 8
    config_ks['dataset_args']['history_max_tokens'] = 384
    config_ks['per_gpu_train_batch_random_sample'] = True
    config_ks["dataset_args"]["negative_sample_method"] = "all"

    config_ks_multitask = config_ks.copy()
    config_ks_multitask["task"] = "multitask"
    train_joint_selection_types = ["domain", "entity", "doc"]
    f_string = "_".join(train_joint_selection_types)
    config_ks_multitask["dataset_args"]["train_joint_selection_types"] = train_joint_selection_types
        
    train_ks_model_multitask = TrainBaselineJob(
        'ks_baseline_no_domain', 
        config_ks_multitask,
        gpus=4, 
        extra_time=30, 
        additional_args="--multitask --selection"
    )
    preds = run_stages(
        train_ks_model_multitask.checkpoint, 
        train_joint_selection_types, 
        "--eval_only --eval_all_snippets --multitask --selection --eval_task \"selection\"", 
        labels=ground_truth_labels
    )
    score_ks_model_job = ScoreBaselineJob(preds)
    tk.register_output(f'ks_hierarchical_multitask_{f_string}.json', preds)
    tk.register_output(f'ks_hierarchical_multitask_{f_string}.score.json', score_ks_model_job.scores)

    for dataset, ground_truth, data_root in zip(['val', 'test'], [ground_truth_labels, eval_ground_truth], [data_root, data_eval_root]):
        # Run on test data
        eval_args = {
            'data_path': data_root,
            'eval_split': dataset,
            'labels': ground_truth
        }

        preds = run_stages(
            train_ks_model_multitask.checkpoint, 
            train_joint_selection_types,
            "--eval_only --eval_all_snippets --multitask --selection --eval_task \"selection\"", 
            **eval_args
        )
        tk.register_output(f'ks_hierarchical_multitask_{f_string}_{dataset}.json', preds)
        score_ks_model_job = ScoreBaselineJob(preds, data_path=data_eval_root, dataset="test")
        tk.register_output(f'ks_hierarchical_multitask_{f_string}_{dataset}.score.json', score_ks_model_job.scores)

    config_ks["dataset_args"]["train_joint_selection_types"] = ["domain", "entity", "doc"]
    config_ks["dataset_args"]["selection_type"] = "domain"
    train_job_domain = TrainBaselineJob(f"ks_domain", config_ks, gpus=1, extra_time=2, additional_args="--skip_per_epoch_eval")

    config_ks["dataset_args"]["selection_type"] = "doc"
    train_job_doc = TrainBaselineJob(f"ks_doc", config_ks, gpus=1, extra_time=2, additional_args="--skip_per_epoch_eval")
    
    config_ks["dataset_args"]["selection_type"] = "entity"
    train_job_entity = TrainBaselineJob(f"ks_entity", config_ks, gpus=1, extra_time=2, additional_args="--skip_per_epoch_eval")


    for dataset, ground_truth, data_root in zip(['val', 'test'], [ground_truth_labels, eval_ground_truth], [data_root, data_eval_root]):
        # Run on test data
        eval_args = {
            'data_path': data_root,
            'eval_split': dataset,
            'labels': ground_truth
        }

        preds = run_stages(
            [train_job_doc, train_job_domain, train_job_entity], 
            "--eval_only --eval_all_snippets", 
            **eval_args
        )
        tk.register_output(f'ks_hierarchical_{f_string}_{dataset}.json', preds)
        score_ks_model_job = ScoreBaselineJob(preds, data_path=data_root, dataset=dataset)
        tk.register_output(f'ks_hierarchical_{f_string}_{dataset}.score.json', score_ks_model_job.scores)
