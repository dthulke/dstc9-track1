import argparse
import glob
import logging
import os
import random
import shutil
import json
import sys

from typing import Dict, List, Tuple
from argparse import Namespace

import numpy as np
import sklearn
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    GPT2DoubleHeadsModel,
    GPT2PreTrainedModel,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    BartConfig,
)

from baseline.dataset import (
    ResponseGenerationDataset,
    KnowledgeSelectionDataset,
    KnowledgeTurnDetectionDataset,
    EmbeddingDataset,
    SPECIAL_TOKENS,
    init_special_tokens_by_model,
)
from baseline.models import GPT2ClsDoubleHeadsModel, GPT2ForSequenceClassificationModel, BartForSequenceEmbedding, RobertaForSequenceEmbedding
from baseline.utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args
)
from baseline.utils.model import (
    run_batch_detection,
    run_batch_generation,
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_embedding_eval,
)
from baseline.utils.data import write_selection_preds, write_detection_preds

from baseline.transformers.configuration_rag import RagConfig
from baseline.transformers.modeling_rag import RagToken


def embed_knowledge(args, model, knowledge_dataset):
    knowledge_sampler = SequentialSampler(knowledge_dataset) if args.local_rank == -1 else DistributedSampler(knowledge_dataset, shuffle=False)
    knowledge_dataloader = DataLoader(
        knowledge_dataset,
        sampler=knowledge_sampler,
        batch_size=1, # only support batch_size=1 for sampling right now
        collate_fn=knowledge_dataset.collate_fn
    )

    knowledge_embeddings = []
    knowledge_infos = []
    model.eval()
    for batch in tqdm(knowledge_dataloader, desc="Knowledge Embeddings", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            data_infos = batch[-1]
            embeddings = run_batch_embedding_eval(args, model, batch)
            
            knowledge_embeddings.append(embeddings)
            knowledge_infos.append(data_infos)

    return knowledge_embeddings, knowledge_infos

