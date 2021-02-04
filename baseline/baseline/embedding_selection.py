import argparse
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

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, GPT2PreTrainedModel, BartConfig

from .dataset import KnowledgeEmbeddingDataset, DialogEmbeddingDataset, SPECIAL_TOKENS, init_special_tokens_by_model
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from .utils.model import run_batch_embedding_eval
from .utils.metrics import (
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE
)
from .utils.data import write_selection_preds
from .main import get_model_class, update_args_by_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, knowledge_dataset, dialog_dataset, model, tokenizer, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    knowledge_sampler = SequentialSampler(knowledge_dataset)
    knowledge_dataloader = DataLoader(
        knowledge_dataset,
        sampler=knowledge_sampler,
        batch_size=1, # only support batch_size=1 for sampling right now
        collate_fn=knowledge_dataset.collate_fn
    )

    args.tokenizer = tokenizer
    knowledge_embeddings = []
    knowledge_infos = []
    do_evaluate = False
    model.eval()
    for batch in tqdm(knowledge_dataloader, desc="Knowledge Embeddings", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            data_infos = batch[-1]
            embeddings = run_batch_embedding_eval(args, model, batch)
            
            knowledge_embeddings.append(embeddings)
            knowledge_infos.append(data_infos)

    knowledge_embeddings = torch.cat(knowledge_embeddings)

    dialog_sampler = SequentialSampler(dialog_dataset)
    dialog_dataloader = DataLoader(
        dialog_dataset,
        sampler=dialog_sampler,
        batch_size=1, # only support batch_size=1 for sampling right now
        collate_fn=dialog_dataset.collate_fn
    )

    data_infos = []
    sorted_pred_keys = []
    label_keys = []
    for batch in tqdm(dialog_dataloader, desc="Dialog Embeddings", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            data_info = batch[-1]
            embeddings = run_batch_embedding_eval(args, model, batch)

            batch_size, dim = embeddings.shape
            assert batch_size == 1
            
            smaller_is_better = True
            if args.embedding_loss == 'nll':
                smaller_is_better = False
                dists = torch.matmul(knowledge_embeddings, embeddings[0].view(-1, 1)).view(-1)
            else:
                dists = torch.norm(knowledge_embeddings - embeddings[0], dim=1)
            sorted_keys = [(info['key'][0], dist) for info, dist in sorted(zip(knowledge_infos, dists), key=lambda x: x[1], reverse=not smaller_is_better)]
            sorted_pred_keys.append(sorted_keys)
            label_keys.append(data_info['knowledge_key'][0])
            data_infos.append(data_info)


    result = dict()
    if args.local_rank in [-1, 0]:
        label_keys = np.array(label_keys).reshape(-1)
        all_pred_keys = np.array([sorted_keys[0][0] for sorted_keys in sorted_pred_keys])
        accuracy = np.sum(all_pred_keys == label_keys) / len(label_keys)
        result = {"loss": 0.0, "accuracy": accuracy}
        if args.output_file:
            write_selection_preds(dialog_dataset.dataset_walker, args.output_file, data_infos, sorted_pred_keys, topk=100)

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument('--embedding', action='store_true')
    parser.add_argument("--dataroot", type=str, default="",
                        help="Path to dataset, will override the path in config.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="If set, the labels will be loaded not from the default path, but from this file instead."
                        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.")    
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)
    
    args.params = params # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    model_class = get_model_class(args)

    args.output_dir = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    init_special_tokens_by_model(tokenizer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    model = model_class.from_pretrained(args.checkpoint)
    model.to(args.device)

    update_args_by_model(args, dataset_args, model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Embedding selection parameters %s", args)
    
    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        knowledge_dataset = KnowledgeEmbeddingDataset(dataset_args, tokenizer, split_type=args.eval_dataset, labels_file=args.labels_file)
        dialog_dataset = DialogEmbeddingDataset(dataset_args, tokenizer, split_type=args.eval_dataset, labels_file=args.labels_file)
        result = evaluate(args, knowledge_dataset, dialog_dataset, model, tokenizer, desc=args.eval_desc or "val")

    return result


if __name__ == "__main__":
    main()
