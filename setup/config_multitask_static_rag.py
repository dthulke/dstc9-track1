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

from recipe.baseline import TrainBaselineJob, RunBaselineJob, ScoreBaselineJob, RunMultitaskOnTrainJob
from recipe.data import MakeTrueTargets, CreateOutOfDomainTestData, ConcatLabels
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
    config_decoding['no_repeat_ngram_size'] = 4

    config_decoding_tuned = copy.deepcopy(config_decoding)
    config_decoding_tuned['repetition_penalty'] = 1.2

    config_decoding_tuned_final = copy.deepcopy(config_decoding_tuned)
    config_decoding_tuned_final['max_length'] = 60

    config_decoding_tuned_final_small_beam = copy.deepcopy(config_decoding_tuned_final)
    config_decoding_tuned_final_small_beam['num_beams'] = 4

    base_model_name_or_path = 'facebook/bart-large'
    config_ktd['model_name_or_path'] = base_model_name_or_path
    config_ks['model_name_or_path'] = base_model_name_or_path
    config_rg['model_name_or_path'] = base_model_name_or_path

    learning_rate = 6.25e-6
    config_ktd['learning_rate'] = learning_rate
    config_ks['learning_rate'] = learning_rate
    config_rg['learning_rate'] = learning_rate

    # Eval results
    data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'
    test_multitask_detection_labels = '/work/smt2/daheim/setups/sisyphus/dstc9-track1/work/baseline/RunBaselineJob.IN0WA82PDUlb/output/preds.json'

    train_selection_labels = RunMultitaskOnTrainJob(labels=data_root / 'train' / 'labels.json', data_path=data_root, eval_split='train', split=64, splits_to_run=[2, 30, 31])

    # RAG Model
    initial_rag_checkpoint = CreateRagCheckpoint('', 'facebook/bart-large').checkpoint

    rag_static_config = copy.deepcopy(config_rg)
    rag_static_config['dataset_args']['history_max_tokens'] = 128
    rag_static_config['embedding_loss'] = 'nll'
    rag_static_config['force_correct_knowledge_in_training'] = True
    rag_static_config['model_name_or_path'] = initial_rag_checkpoint
    rag_static_config['is_rag_model'] = True
    rag_static_config['knowledge_encoder_checkpoint'] = ''
    rag_static_config['selection_labels'] = train_selection_labels.preds
    rag_static_config['per_gpu_train_batch_size'] = 1
    rag_static_config['per_gpu_eval_batch_size'] = 4
    rag_static_config['gradient_accumulation_steps'] = 16

    bart_train_rg = TrainBaselineJob('rag_static_multitask', rag_static_config, gpus=1)

    # Run on val
    val_selection_labels = '/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/ScoreBaselineJob.PHdoejms41pM/input/baseline_RunBaselineJob.dWFOJbttXJoH/output/preds.json'
    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=val_selection_labels, config=config_decoding, additional_args=f"--selection_labels {val_selection_labels} --generate --generation_params_file params.json")
    tk.register_output(f'rag_static_multitask.json', run_rg.preds)
    score_job = ScoreBaselineJob(run_rg.preds)
    tk.register_output('rag_static_multitask.score.json', score_job.scores)

    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=val_selection_labels, config=config_decoding_tuned, additional_args=f"--selection_labels {val_selection_labels} --generate --generation_params_file params.json")
    tk.register_output(f'rag_static_multitask_tuned.json', run_rg.preds)
    score_job = ScoreBaselineJob(run_rg.preds)
    tk.register_output('rag_static_multitask_tuned.score.json', score_job.scores)

    # Run on eval
    eval_results = '/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/RunBaselineJob.QqsZSuzEj9qh/output/preds.json'
    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
    tk.register_output(f'rag_static_multitask.test.json', run_rg.preds)

    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding_tuned, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
    tk.register_output(f'rag_static_multitask_tuned.test.json', run_rg.preds)

    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding_tuned_final, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
    tk.register_output(f'rag_static_multitask_tuned_final.test.json', run_rg.preds)
    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding_tuned_final_small_beam, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
    tk.register_output(f'rag_static_multitask_tuned_final_small_beam.test.json', run_rg.preds)

    with tk.block('full-training-data'):
        data_root_full = pathlib.Path(gs.BASELINE_ROOT) / 'data_full'

        full_selection_labels = ConcatLabels(train_selection_labels.preds, '/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/ScoreBaselineJob.PHdoejms41pM/input/baseline_RunBaselineJob.dWFOJbttXJoH/output/preds.json')

        full_rag_static_config = copy.deepcopy(rag_static_config)
        full_rag_static_config['selection_labels'] = full_selection_labels.labels

        bart_train_rg = TrainBaselineJob('rag_static_multitask_full_data', full_rag_static_config, data_path=data_root_full, gpus=1)

        # Run on val
        val_selection_labels = '/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/ScoreBaselineJob.PHdoejms41pM/input/baseline_RunBaselineJob.dWFOJbttXJoH/output/preds.json'
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=val_selection_labels, config=config_decoding, additional_args=f"--selection_labels {val_selection_labels} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_multitask_full_data.json', run_rg.preds)
        score_job = ScoreBaselineJob(run_rg.preds)
        tk.register_output('rag_static_multitask_full_data.score.json', score_job.scores)

        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=val_selection_labels, config=config_decoding_tuned, additional_args=f"--selection_labels {val_selection_labels} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_multitask_full_data_tuned.json', run_rg.preds)
        score_job = ScoreBaselineJob(run_rg.preds)
        tk.register_output('rag_static_multitask_full_data_tuned.score.json', score_job.scores)

        # Run on eval
        eval_results = '/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/RunBaselineJob.QqsZSuzEj9qh/output/preds.json'
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_multitask_full_data.test.json', run_rg.preds)

        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding_tuned, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_multitask_full_data_tuned.test.json', run_rg.preds)

    # Run normal generation
    train_rg_model = TrainBaselineJob('rg_baseline', config_rg)

    run_rg_model = RunBaselineJob(train_rg_model.checkpoint, labels=val_selection_labels, config=config_decoding_tuned, additional_args=f"--generate --generation_params_file params.json")
    tk.register_output(f'rg_baseline_multitask_selection_tuned.json', run_rg_model.preds)
    score_rg_model_job = ScoreBaselineJob(run_rg_model.preds)
    tk.register_output('rg_baseline_multitask_selection_tuned.score.json', score_rg_model_job.scores)

    # Run on eval
    eval_results = '/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/RunBaselineJob.QqsZSuzEj9qh/output/preds.json'
    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding_tuned_final, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
    tk.register_output(f'rg_baseline_multitask_selection_tuned_final.test.json', run_rg.preds)
    run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=eval_results, config=config_decoding_tuned_final_small_beam, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {eval_results} --generate --generation_params_file params.json")
    tk.register_output(f'rg_baseline_multitask_selection_tuned_final_small_beam.test.json', run_rg.preds)
