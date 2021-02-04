import argparse
import copy
import glob
import logging
import os
import random
import shutil
import json

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2DoubleHeadsModel,
    GPT2PreTrainedModel,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    BartConfig,
)

from .dataset import (
    ResponseGenerationDataset,
    KnowledgeSelectionDataset,
    KnowledgeTurnDetectionDataset,
    MultiTaskDataset,
    SPECIAL_TOKENS,
    init_special_tokens_by_model,
    JointSelectionDataset
)
from .models import (
    BartForJointSelection,
    BartForMultitaskModeling,
    GPT2ClsDoubleHeadsModel,
    GPT2ForSequenceClassificationModel,
    GPT2MultiTask,
    RobertaForMultitaskModeling
)
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import (
    run_batch_detection,
    run_batch_generation,
    run_batch_multitask_train,
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_joint_selection_train
)
from .utils.data import write_selection_preds, write_detection_preds
from .main import update_args_by_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def get_model_class(args):
    if args.selection:
        return BartForJointSelection
    else:
        if 'gpt' in args.model_name_or_path:
            return GPT2MultiTask
        elif 'roberta' in args.model_name_or_path:
            return RobertaForMultitaskModeling
        elif 'bart' in args.model_name_or_path:
            return BartForMultitaskModeling
        else:
            raise NotImplementedError()

def _align_namespace(namespace, task, selection_type):
    namespace = copy.deepcopy(namespace)
    namespace.task = task
    namespace.selection_type = selection_type
    return namespace


def get_classes(args, task):
    model_class = get_model_class(args)
    if task == "multitask":
        if args.selection:
            return JointSelectionDataset, model_class, run_batch_joint_selection_train, run_batch_selection_eval
        else:
            return MultiTaskDataset, model_class, run_batch_multitask_train, run_batch_selection_eval
    elif task.lower() == "generation":
        return ResponseGenerationDataset, model_class, run_batch_generation, run_batch_generation
    elif task.lower() == "selection":
        return KnowledgeSelectionDataset, model_class, run_batch_selection_train, run_batch_selection_eval
    elif task.lower() == "detection":
        return KnowledgeTurnDetectionDataset, model_class, run_batch_detection, run_batch_detection
    else:
        raise ValueError("args.task not in ['multitask', 'generation', 'selection', 'detection'], got %s" % args.task)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval, dataset_args) -> Tuple[int, float]:
    logger.info(args)
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    global_step = 0
    previous_checkpoint = None
    model.zero_grad()
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # for reproducibility

    for current_epoch in train_iterator:
        local_steps = 0
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_steps = len(epoch_iterator)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            losses, _, _ = run_batch_fn_train(args, model, batch)

            loss = sum(losses)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == total_steps:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss/local_steps)

        if args.selection:
            if current_epoch in [0, 9]:
                for selection_type in args.dataset_args.train_joint_selection_types:
                    dataset_class, _, _, run_batch_fn_eval = get_classes(args, "selection")
                    eval_dataset = dataset_class(_align_namespace(dataset_args, "selection", selection_type), tokenizer, split_type="val")
                    results = evaluate(_align_namespace(args, "selection", selection_type), "selection", eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
            else:
                results = {key: .0 for key in ["loss", "accuracy", "precision", "recall"]}

        else:
            for task in ["generation", "selection", "detection"]:
                dataset_class, _, _, run_batch_fn_eval = get_classes(args, task)
                eval_dataset = dataset_class(dataset_args, tokenizer, split_type="val")
                results = evaluate(args, task, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))

        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)

            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
            logger.info("Saving model checkpoint to %s", output_dir)

            if previous_checkpoint is not None:
                # remove previous checkpoint if exists
                if os.path.isdir(previous_checkpoint):
                    shutil.rmtree(previous_checkpoint)

            previous_checkpoint = output_dir

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / local_steps


def evaluate(args, task, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle variable number of candidates
    if task == "selection":
        args.eval_batch_size = 1
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (task != "selection" or eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            loss, _, mc_logits, mc_labels = run_batch_fn(args, model, batch)
            if task == "detection":
                mc_logits = mc_logits.sigmoid()
            if task in ["selection", "detection"]:
                data_infos.append(batch[-1])
            all_preds.append(mc_logits.detach().cpu().numpy())
            all_labels.append(mc_labels.detach().cpu().numpy())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    # Synchronize eval results between the nodes
    # TODO there is probably a better method than dumping everything to the filesystem :)
    if args.local_rank != -1:
        import pickle
        if args.local_rank != 0:
            # Store results
            process_results = eval_loss, nb_eval_steps, data_infos, all_preds, all_labels
            with open(f'subprocess_{args.local_rank}_result.pickle', 'wb') as fp:
                pickle.dump(process_results, fp)
            torch.distributed.barrier()
            return {}

        if args.local_rank == 0:
            # Block until all other processes finished
            torch.distributed.barrier()
            for i in range(1, torch.distributed.get_world_size()):
                with open(f'subprocess_{i}_result.pickle', 'rb') as fp:
                    process_results = pickle.load(fp)
                    eval_loss += process_results[0]
                    nb_eval_steps += process_results[1]
                    data_infos += process_results[2]
                    all_preds += process_results[3]
                    all_labels += process_results[4]

    eval_loss = eval_loss / nb_eval_steps

    if task.lower() == "generation":
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {"perplexity": perplexity, "loss": eval_loss}
    elif task.lower() == "selection":
        all_labels = np.array(all_labels).reshape(-1)
        all_pred_ids = np.array([np.argmax(logits) for logits in all_preds])
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        logger.info("Avg. # of candidates: %f", sum([len(arr[0]) for arr in all_preds]) / len(all_preds))
        result = {"loss": eval_loss, "accuracy": accuracy}
        if args.output_file:
            scores = [logits.squeeze() for logits in all_preds]
            sorted_pred_ids = [np.argsort(score)[::-1] for score in scores]
            write_selection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_ids, scores, topk=100)
    elif task.lower() == "detection":
        all_labels = np.concatenate(all_labels)
        all_pred_ids = (np.concatenate(all_preds) > 0.5)
        accuracy = np.sum(all_pred_ids == all_labels) / len(all_labels)
        precision = sklearn.metrics.precision_score(all_labels, all_pred_ids)
        recall = sklearn.metrics.recall_score(all_labels, all_pred_ids)
        result = {"loss": eval_loss, "accuracy": accuracy, "precision": precision, "recall": recall}
        if args.output_file:
            write_detection_preds(eval_dataset.dataset_walker, args.output_file, data_infos, all_pred_ids)
    else:
        raise ValueError("args.task not in ['generation', 'selection', 'detection'], got %s" % task)

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument('--multitask', action='store_true')
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--no_labels", action="store_true",
                        help="Read a dataset without labels.json. This option is useful when running "
                             "knowledge-seeking turn detection on test dataset where labels.json is not available.")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="", help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--eval_task", type=str, default=None,
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--per_gpu_train_batch_random_sample", action='store_true')
    parser.add_argument('--selection', action='store_true')
    parser.add_argument("--selection_type", type=str, default=None,
                        help="Optional to specify evaluation task in multitask setting.")

    args, additional_args = parser.parse_known_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)
    
    args.multitask = True
    args.params = params # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.multitask = True

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0") #if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    logger = logging.getLogger(__name__)
    logger.info("Running on {} GPU(s)".format(args.n_gpu))

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.eval_only and args.eval_task is not None:
        args.task = args.eval_task
        dataset_args.task = args.eval_task
        if args.selection_type is not None:
            dataset_args.selection_type = args.selection_type
    
    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(args, args.task)

    if args.eval_only:
        args.output_dir = args.checkpoint
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        init_special_tokens_by_model(tokenizer)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        config.num_labels = 1
        if args.selection:
            config.train_joint_selection_types = args.dataset_args.train_joint_selection_types
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        init_special_tokens_by_model(tokenizer)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    update_args_by_model(args, dataset_args, model)

    logger.info("Training/evaluation parameters %s", args)
    if not args.eval_only:
        train_dataset = dataset_class(dataset_args, tokenizer, split_type="train")
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type="val")

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval, dataset_args)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            model.to(args.device)

    # Evaluation
    if args.eval_only:
        eval_dataset = dataset_class(dataset_args, tokenizer, split_type=args.eval_dataset, labels=not args.no_labels, labels_file=args.labels_file)
        result = evaluate(args, args.task, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or "val")
    else:
        result = {"loss": .0, "accuracy": .0, "precision": .0, "recall": .0}

    return result


if __name__ == "__main__":
    main()
