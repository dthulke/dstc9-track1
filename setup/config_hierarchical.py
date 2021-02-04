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

    config_ks_domain_entity = copy.deepcopy(config_ks)
    config_ks_domain_entity['dataset_args']['selection_type'] = 'domain_entity'
    train_ks_domain_entity_model = TrainBaselineJob('ks_baseline_domain-entity', config_ks_domain_entity, extra_time=12)

    run_ks_domain_entity_model = RunBaselineJob(train_ks_domain_entity_model.checkpoint, labels=run_ktd_model.preds, gpus=1, additional_args="--eval_only --eval_all_snippets", extra_time=4)
    score_ks_domain_entity_model_job = ScoreBaselineJob(run_ks_domain_entity_model.preds)
    tk.register_output(f'ks_baseline_domain-entity_ktd_baseline.json', run_ks_domain_entity_model.preds)
    tk.register_output(f'ks_baseline_domain-entity_ktd_baseline.score.json', score_ks_domain_entity_model_job.scores)

    config_ks_doc = copy.deepcopy(config_ks)
    config_ks_doc['dataset_args']['selection_type'] = 'doc'
    train_ks_doc_model = TrainBaselineJob('ks_baseline_doc', config_ks_doc, extra_time=12)

    run_ks_doc_model = RunBaselineJob(train_ks_doc_model.checkpoint, labels=run_ks_domain_entity_model.preds, gpus=1, additional_args="--eval_only --eval_all_snippets", extra_time=4)
    score_ks_doc_model_job = ScoreBaselineJob(run_ks_doc_model.preds)
    tk.register_output(f'ks_baseline_domain-entity_doc_ktd_baseline.json', run_ks_doc_model.preds)
    tk.register_output(f'ks_baseline_domain-entity_doc_ktd_baseline.score.json', score_ks_doc_model_job.scores)

    # Run on val labels
    run_ks_domain_entity_model_val_labels = RunBaselineJob(train_ks_domain_entity_model.checkpoint, labels=data_root / 'val' / 'labels.json', gpus=1, additional_args="--eval_only --eval_all_snippets", extra_time=4)
    score_ks_domain_entity_model_job = ScoreBaselineJob(run_ks_domain_entity_model_val_labels.preds)
    tk.register_output(f'ks_baseline_domain-entity.json', run_ks_domain_entity_model_val_labels.preds)
    tk.register_output(f'ks_baseline_domain-entity.score.json', score_ks_domain_entity_model_job.scores)

    run_ks_doc_model_val_labels = RunBaselineJob(train_ks_doc_model.checkpoint, labels=run_ks_domain_entity_model_val_labels.preds, gpus=1, additional_args="--eval_only --eval_all_snippets", extra_time=4)
    score_ks_doc_model_job = ScoreBaselineJob(run_ks_doc_model_val_labels.preds)
    tk.register_output(f'ks_baseline_domain-entity_doc.json', run_ks_doc_model_val_labels.preds)
    tk.register_output(f'ks_baseline_domain-entity_doc.score.json', score_ks_doc_model_job.scores)


    with tk.block('run_on_eval'):
        data_eval_root = pathlib.Path(gs.BASELINE_ROOT) / 'data_eval'
        test_multitask_detection_labels = '/work/smt2/daheim/setups/sisyphus/dstc9-track1/work/baseline/RunBaselineJob.IN0WA82PDUlb/output/preds.json'
        
        run_ks_domain_entity_model = RunBaselineJob(train_ks_domain_entity_model.checkpoint, labels=test_multitask_detection_labels, gpus=1, data_path=data_eval_root, eval_split='test', additional_args="--eval_only --eval_all_snippets", split=2)
        tk.register_output(f'ks_baseline_domain-entity.test.json', run_ks_domain_entity_model.preds)
        run_ks_doc_test_model = RunBaselineJob(train_ks_doc_model.checkpoint, labels=run_ks_domain_entity_model.preds, gpus=1, data_path=data_eval_root, eval_split='test', additional_args="--eval_only --eval_all_snippets", split=1)
        tk.register_output(f'ks_baseline_domain-entity_doc.test.json', run_ks_doc_test_model.preds)


    with tk.block('rag_static'):
        train_labels = str(data_root / 'train' / 'labels.json')
        run_ks_domain_entity_model = RunBaselineJob(train_ks_domain_entity_model.checkpoint, labels=train_labels, gpus=1, eval_split='train', additional_args="--eval_only --eval_all_snippets", extra_time=4, split=4)
        tk.register_output(f'ks_baseline_domain-entity.train.json', run_ks_domain_entity_model.preds)
        run_ks_doc_train_model = RunBaselineJob(train_ks_doc_model.checkpoint, labels=run_ks_domain_entity_model.preds, gpus=1, eval_split='train', additional_args="--eval_only --eval_all_snippets", extra_time=4, split=4)
        tk.register_output(f'ks_baseline_domain-entity_doc.train.json', run_ks_doc_train_model.preds)

        initial_rag_checkpoint = CreateRagCheckpoint('', 'facebook/bart-large').checkpoint

        rag_static_config = copy.deepcopy(config_rg)
        rag_static_config['dataset_args']['history_max_tokens'] = 128
        rag_static_config['embedding_loss'] = 'nll'
        rag_static_config['force_correct_knowledge_in_training'] = True
        rag_static_config['model_name_or_path'] = initial_rag_checkpoint
        rag_static_config['is_rag_model'] = True
        rag_static_config['knowledge_encoder_checkpoint'] = ''
        rag_static_config['selection_labels'] = run_ks_doc_train_model.preds
        rag_static_config['per_gpu_train_batch_size'] = 1
        rag_static_config['per_gpu_eval_batch_size'] = 4
        rag_static_config['gradient_accumulation_steps'] = 16

        bart_train_rg = TrainBaselineJob('rag_static_individual_selection', rag_static_config, gpus=1)

        # Run on val
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_model.preds, config=config_decoding, additional_args=f"--selection_labels {run_ks_doc_model.preds} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_individual_selection.json', run_rg.preds)
        score_job = ScoreBaselineJob(run_rg.preds)
        tk.register_output('rag_static_individual_selection.score.json', score_job.scores)

        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_model.preds, config=config_decoding_tuned, additional_args=f"--selection_labels {run_ks_doc_model.preds} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_individual_selection_tuned.json', run_rg.preds)
        score_job = ScoreBaselineJob(run_rg.preds)
        tk.register_output('rag_static_individual_selection_tuned.score.json', score_job.scores)

        # Run on eval
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_test_model.preds, config=config_decoding, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {run_ks_doc_test_model.preds} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_individual_selection.test.json', run_rg.preds)
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_test_model.preds, config=config_decoding_tuned, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {run_ks_doc_test_model.preds} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_individual_selection_tuned.test.json', run_rg.preds)
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_test_model.preds, config=config_decoding_tuned_final, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {run_ks_doc_test_model.preds} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_individual_selection_tuned_final.test.json', run_rg.preds)
        run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_test_model.preds, config=config_decoding_tuned_final_small_beam, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {run_ks_doc_test_model.preds} --generate --generation_params_file params.json")
        tk.register_output(f'rag_static_individual_selection_tuned_final_small_beam.test.json', run_rg.preds)

        with tk.block('full-training-data'):
            data_root_full = pathlib.Path(gs.BASELINE_ROOT) / 'data_full'

            full_selection_labels = ConcatLabels(run_ks_doc_train_model.preds, run_ks_doc_model_val_labels.preds)

            full_rag_static_config = copy.deepcopy(rag_static_config)
            full_rag_static_config['selection_labels'] = full_selection_labels.labels

            bart_train_rg = TrainBaselineJob('rag_static_individual_selection_full_data', full_rag_static_config, data_path=data_root_full, gpus=1)

            # Run on val
            run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_model.preds, config=config_decoding, additional_args=f"--selection_labels {run_ks_doc_model.preds} --generate --generation_params_file params.json")
            tk.register_output(f'rag_static_individual_selection_full_data.json', run_rg.preds)
            score_job = ScoreBaselineJob(run_rg.preds)
            tk.register_output('rag_static_individual_selection_full_data.score.json', score_job.scores)

            run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_model.preds, config=config_decoding_tuned, additional_args=f"--selection_labels {run_ks_doc_model.preds} --generate --generation_params_file params.json")
            tk.register_output(f'rag_static_individual_selection_full_data_tuned.json', run_rg.preds)
            score_job = ScoreBaselineJob(run_rg.preds)
            tk.register_output('rag_static_individual_selection_full_data_tuned.score.json', score_job.scores)

            # Run on eval
            run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_test_model.preds, config=config_decoding, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {run_ks_doc_test_model.preds} --generate --generation_params_file params.json")
            tk.register_output(f'rag_static_individual_selection_full_data.test.json', run_rg.preds)

            run_rg = RunBaselineJob(bart_train_rg.checkpoint, labels=run_ks_doc_test_model.preds, config=config_decoding_tuned, data_path=data_eval_root, eval_split='test', additional_args=f"--selection_labels {run_ks_doc_test_model.preds} --generate --generation_params_file params.json")
            tk.register_output(f'rag_static_individual_selection_full_data_tuned.test.json', run_rg.preds)


    # Generation
    train_rg_model = TrainBaselineJob('rg_baseline', config_rg)
    run_rg_model = RunBaselineJob(train_rg_model.checkpoint, labels=run_ks_doc_model.preds, config=config_decoding, additional_args=f"--generate --generation_params_file params.json")
    tk.register_output(f'rg_baseline_domain-entity_doc.json', run_rg_model.preds)
    score_rg_model_job = ScoreBaselineJob(run_rg_model.preds)
    tk.register_output('rg_baseline_domain-entity_doc.score.json', score_rg_model_job.scores)
