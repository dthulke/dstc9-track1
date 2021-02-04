import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BartForSequenceClassification
from tqdm import tqdm

from baseline.dataset import (
    SPECIAL_TOKENS,
    init_special_tokens_by_model,
)
from baseline.main import get_classes, update_args_by_model
from baseline.models import (
    BartForSequenceEmbedding, 
    RobertaForSequenceEmbedding
)
from baseline.utils.argument import (
    set_default_params,
    set_default_dataset_params,
    verify_args
)
from baseline.utils.model import *

MODELS = {
    "embedding": {
        "facebook/bart": BartForSequenceEmbedding,
        "roberta": RobertaForSequenceEmbedding
    },
    "selection": {
        "facebook/bart": BartForSequenceClassification
    }
}


class Timer(object):
    """
    Implements common functionality to measure the execution time of torch
    models in a base class, which can be extended for different types
    of learning problems.

    Examples:
    >>> from runtimes import Timer

    >>> timer = Timer(model)

    >>> time = timer.measure_forward_time(**forward_args)
    """


    def __init__(self, model):
        self._model = model

    @torch.no_grad()
    def measure_forward_time(self, forward_args):
        """Measures the time it takes to forward the given input."""
        start = time.time()
        self._model(forward_args)
        return time.time() - start

    @torch.no_grad()
    def time(self, args, dataset, num_samples, **kwargs):
        """
        Times {num_samples} from the given {dataset}.
        
        Args:
            args: Namespace containing relevant parameters.
            dataset: torch.utils.data.Dataset
            num_samples: Number of samples to use for assessment.

        Returns:
            list of execution times with length {num_samples}.
        """ 
        sampler = torch.utils.data.RandomSampler(
            dataset,
            num_samples=num_samples,
            replacement=True
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            sampler=sampler,
            batch_size=1,
            collate_fn=dataset.collate_fn,
        )        

        times = []
        for batch in tqdm(data_loader, desc="Iteration"):
            times.append(self.time_batch(args, batch, **kwargs))
        
        return times

    def time_batch(self, args, batch, **kwargs):
        """
        Skeleton for working off one batch with the given parameters.
        
        Args:
            args: Namespace containing relevant parameters.
            batch: Batch to work off.
 
        """
        pass

class TimerForSequenceClassification(Timer):
    """
    Implements the assessment of execution times for sequence classification
    for knowledge-selection.

    Examples:
    >>> from runtimes import TimerForSequenceClassification

    >>> timer = TimerForSequenceClassification(model)

    >>> times = timer.time(args, dataset, num_samples)
    """
    
    def time_batch(self, args, batch):
        start = time.time()

        args.eval_batch_size = 1
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch 
                        if isinstance(input_tensor, torch.Tensor))
        input_ids, token_type_ids, mc_token_ids, _, _ = batch
        all_mc_logits = []
        input_size = input_ids.shape[2]

        num_candidates = args.max_candidates_per_forward_eval

        for index in range(0, input_ids.size(1), num_candidates): 
            
            forward_args = {
                'input_ids': input_ids[0, index: index+num_candidates] \
                    .unsqueeze(1).view(-1, input_size),
            }
            if args.type_vocab_size == args.vocab_size:
                # ID of the cls label
                forward_args['mc_token_ids'] = mc_token_ids[0, index: index+num_candidates] \
                    .unsqueeze(1).view(-1)
            if args.type_vocab_size > 0 and token_type_ids is not None:
                forward_args['token_type_ids'] = token_type_ids[0, index: index+num_candidates] \
                    .unsqueeze(1).view(-1, input_size)
            
            mc_logits = self._model(**forward_args)[0]
            all_mc_logits.append(mc_logits.detach())

        all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)

        scores = [logits.squeeze() for logits in all_mc_logits]
        np.argmax(scores)

        return time.time() - start


TIMER_CLASS = {
    "selection": TimerForSequenceClassification
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset.")
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--knowledge_file", type=str, default="knowledge.json",
                        help="knowledge file name.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to use for execution time measurement.")
    parser.add_argument("--params_file", type=str, default="params.json",
                        help="JSON configuration file")

    args, additional_args = parser.parse_known_args()

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("run.log"),
            streamHandler
        ]
    )
    logger = logging.getLogger(__name__)
    args.eval_only = True

    verify_args(args, parser)

    with open(os.path.join(args.checkpoint, args.params_file), "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = argparse.Namespace(**args)
        dataset_args = argparse.Namespace(**args.dataset_args)
    
    set_default_params(args)
    set_default_dataset_params(dataset_args)
    dataset_args.local_rank = -1
    dataset_args.task = args.task
    args.eval_all_snippets = True
    dataset_args.eval_all_snippets = True
    #args.max_candidates_per_forward_eval = 4
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dataset_class, _, _, _ = get_classes(args)

    prefix = args.model_name_or_path.split("-")[0]
    model_class = MODELS[args.task][prefix]
    model = model_class.from_pretrained(args.checkpoint)
    model.to(args.device)
    update_args_by_model(args, dataset_args, model)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    init_special_tokens_by_model(tokenizer)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    dataset = dataset_class(dataset_args, tokenizer, split_type="val")

    logger.info(args)
    logger.info(dataset_args)

    timer = TIMER_CLASS[args.task](model)

    times = timer.time(args, dataset, args.num_samples)

    logger.info(np.mean(times))

    series = pd.Series(times).describe()

    with open("times.txt", "w") as f:
        for index, value in series.items():
            f.write(f"{index}, {value}")
