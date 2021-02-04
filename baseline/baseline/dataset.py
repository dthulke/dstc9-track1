import os
import copy
import json
import random
import logging
import sys

from itertools import chain

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

from transformers import RobertaTokenizer, BartTokenizer, BertTokenizer
from baseline.transformers.modeling_rag import PreTrainedRagModel, RagModel

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]

# TODO reconsider the specific choice of tokens
def init_special_tokens_by_model(tokenizer):
    #if 'roberta' in model_name.lower() or 'bart' in model_name.lower():

    if issubclass(type(tokenizer), RobertaTokenizer) or issubclass(type(tokenizer), BartTokenizer):
        SPECIAL_TOKENS['bos_token'] = '<s>'
        SPECIAL_TOKENS_VALUES[0] = '<s>'
        SPECIAL_TOKENS['eos_token'] = '</s>'
        SPECIAL_TOKENS_VALUES[1] = '</s>'
        SPECIAL_TOKENS['additional_special_tokens'][3] = '</s>'
        SPECIAL_TOKENS_VALUES[6] = '</s>'
    elif issubclass(type(tokenizer), BertTokenizer):
        SPECIAL_TOKENS['bos_token'] = '[CLS]'
        SPECIAL_TOKENS_VALUES[0] = '[CLS]'
        SPECIAL_TOKENS['eos_token'] = '[SEP]'
        SPECIAL_TOKENS_VALUES[1] = '[SEP]'
        SPECIAL_TOKENS['pad_token'] = '[PAD]'
        SPECIAL_TOKENS_VALUES[2] = '[PAD]'
        SPECIAL_TOKENS['additional_special_tokens'][3] = '[SEP]'
        SPECIAL_TOKENS_VALUES[6] = '[SEP]'


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()

        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.knowledge, self.snippets, self.domains, self.entities = self._prepare_knowledge()

        self._create_examples()

    def _prepare_conversations(self):
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])): # only show progress bar in one process
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs
    
    def _prepare_knowledge(self):
        knowledge = self.knowledge_reader.knowledge
        self.knowledge_docs = self.knowledge_reader.get_doc_list()

        domains = {}
        entities = {}
        tokenized_snippets = dict()
        for snippet in self.knowledge_docs:
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "*", snippet["doc_id"])
            # TODO consider whether there is a better way to integrate the domain, here
            entity_name = (snippet["entity_name"] or "*")
            entity_id = str(snippet["entity_id"]) or "*"
            knowledge = self._knowledge_to_string(snippet["doc"], domain=snippet["domain"], name=entity_name)
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge))
            tokenized_snippets[key] = tokenized_knowledge[:self.args.knowledge_max_tokens]
            if snippet["domain"] not in domains:
                domains[snippet["domain"]] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(snippet["domain"]))
                entities[snippet["domain"]] = {}
            if entity_id not in entities[snippet["domain"]]:
                entities[snippet["domain"]][entity_id] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(snippet["domain"] + self.knowledge_sep_token + entity_name))
        return knowledge, tokenized_snippets, domains, entities

    def _knowledge_to_string(self, doc, name="", domain=""):
        if self.args.include_full_knowledge:
            join_str = " %s " % self.knowledge_sep_token
            return join_str.join([domain, name, doc["title"], doc["body"]])
        return doc["body"]

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=self.args.local_rank not in [-1, 0]):
            dialog_id = dialog["id"]
            label = dialog["label"]
            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue
            
            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            gt_resp = label.get("response", "")
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            # Add speaker tags to the history and response
            # the response is always by speaker2
            # and the current_turn always by speaker1
            truncated_history = [
                [self.speaker1 if (len(truncated_history) - i) % 2 == 1 else self.speaker2] + s
                for i, s in enumerate(truncated_history)
            ]
            tokenized_gt_resp = [self.speaker2] + tokenized_gt_resp

            if target:
                if "knowledge" not in label:
                    # when the labels.json is from knowledge-seeking turn detection,
                    # there will be no ground truth knowledge
                    # so we just use a dummy snippet here
                    if not self.args.eval_all_snippets:
                        raise ValueError("eval_all_snippets is required to be true when taking output from knowledge-seeking turn detection")
                    label["knowledge"] = [self.knowledge_docs[0]]
                elif "entity_id" not in label['knowledge'][0]:
                    if not self.args.eval_all_snippets:
                        raise ValueError("eval_all_snippets is required to be true when taking output from domain prediction")
                    label["knowledge"] = [self.knowledge_reader.get_doc_list(domain=label['knowledge'][0]['domain'])[0]]
                elif "doc_id" not in label['knowledge'][0]:
                    if not self.args.eval_all_snippets:
                        raise ValueError("eval_all_snippets is required to be true when taking output from entity prediction")
                    label["knowledge"] = [self.knowledge_reader.get_doc_list(domain=label['knowledge'][0]['domain'], entity_id=label['knowledge'][0]['entity_id'])[0]]


                knowledge = label["knowledge"][0]
                knowledge_key = "{}__{}__{}".format(knowledge["domain"], knowledge["entity_id"], knowledge["doc_id"])
                # find snippets with same entity as candidates
                prefix = "{}__{}__".format(knowledge["domain"], knowledge["entity_id"])
                knowledge_candidates = [cand for cand in self.snippets.keys() if cand.startswith(prefix)]
                if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                    # if there's not enough candidates during training, we just skip this example
                    if len(knowledge_candidates) < self.args.n_candidates:
                        continue
                used_knowledge = self.snippets[knowledge_key]
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]
            else:
                knowledge_candidates = None
                used_knowledge = []
                knowledge_key = None

            self.examples.append({
                "history": truncated_history[:-1],
                "current_turn": truncated_history[-1],
                "knowledge": used_knowledge,
                "knowledge_key": knowledge_key,
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id
            })

    def build_input_from_segments(self, knowledge, history, current_turn, response):
        """
        Builds a task specific input representation for the current model.
        In general, inputs may consist of the following elements:

        current_turn: the last turn triggering the next response
        knowledge: representation of the knowledge scripts (depending on the output of _knowledge_to_string)
        history: utterances before the current turn
        response: target system response
        """
        raise NotImplementedError
                
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.examples)


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["current_turn"],
            example["response"]
        )
        return instance

    def build_input_from_segments(self, knowledge, history, current_turn, response, decode=False):
        """
        Generates the input for the response generation task.
        It differs between decoder-only models (e.g. GPT-2) and encoder-decoder models (e.g. BART)

        For decoder only models the output is:
        input_ids: <bos> [knowledge] [history] [current_turn] [response] <eos>

        For encoder decoder models:
        input_ids: <bos> [knowledge] [history] [current_turn] <eos>
        decoder_input_ids: <bos> [response] <eos>

        If decode is set to true, the following outputs are generated:

        For decoder only models the output is:
        input_ids: <bos> [knowledge] [history] [current_turn] 

        For encoder decoder models:
        input_ids: <bos> [knowledge] [history] [current_turn] <eos>
        decoder_input_ids: <bos> 
        """
        instance = {}

        # Token type ids are omitted in the generation for now (they are not used in decoding in the baseline)
        if any(model_prefix in self.args.model_name_or_path for model_prefix in ['gpt2', 'roberta',' bert']):
            input_ids = [[self.bos] + knowledge] + history + [current_turn] 
            if not decode:
                input_ids = input_ids + [response + [self.eos]]
            instance['lm_labels'] = ([-100] * sum(len(s) for s in input_ids[:-1])) + [-100] + input_ids[-1][1:]
            instance['input_ids'] = list(chain(*input_ids))
        elif self.args.is_rag_model:
            input_ids = history + [current_turn]
            if not decode:
                output_ids = [[self.bos] + response + [self.eos]]
                # eos is used as padding in BART
                decoder_input_ids = [[self.eos] + [self.bos] + response]
            else:
                output_ids = [[]]
                decoder_input_ids = [[]]
            instance['input_ids'] = list(chain(*input_ids))
            instance["lm_labels"] = list(chain(*output_ids))
            instance['decoder_input_ids'] = list(chain(*decoder_input_ids))
        elif "bart" in self.args.model_name_or_path:
            input_ids = [[self.bos] + knowledge] + history + [current_turn] + [[self.eos]]
            if not decode:
                output_ids = [[self.bos] + response + [self.eos]]
                # eos is used as padding in BART
                decoder_input_ids = [[self.eos] + [self.bos] + response]
            else:
                output_ids = [[]]
                decoder_input_ids = [[]]
            instance['input_ids'] = list(chain(*input_ids))
            instance["lm_labels"] = list(chain(*output_ids))
            instance['decoder_input_ids'] = list(chain(*decoder_input_ids))
        else:
            raise NotImplementedError

        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))

        # Only used in seq2seq models
        decoder_input_ids = torch.tensor([])

        lm_labels = [ins["lm_labels"] for ins in batch]
        if any(model_prefix in self.args.model_name_or_path for model_prefix in ['gpt2', 'roberta',' bert']):
            lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        elif "bart" in self.args.model_name_or_path or self.args.is_rag_model:
            #decoder_input_ids = [ins["decoder_input_ids"] for ins in batch]
            decoder_input_ids = torch.tensor(pad_ids(lm_labels, self.pad))
            from transformers.modeling_bart import shift_tokens_right
            decoder_input_ids = shift_tokens_right(decoder_input_ids, self.pad)

            lm_labels = torch.tensor(pad_ids(lm_labels, -100))
            lm_labels[:, 0] = -100 # Mask prediction of initial bos token
        else:
            raise NotImplementedError

        return input_ids, decoder_input_ids, lm_labels


class ResponseGenerationEvalDataset(ResponseGenerationDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch


class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)
        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

    def _knowledge_to_string(self, doc, name="", domain=""):
        join_str = " %s " % self.knowledge_sep_token
        if self.args.selection_type == "domain":
            return domain
        elif self.args.selection_type == "entity" or self.args.selection_type == "domain_entity":
            return join_str.join([domain, name])
        return join_str.join([domain, name, doc["title"], doc["body"]])

    def _split_int_array(self, seq, smallest):    
        group = []    
        for num in seq:
            if num != smallest:
                group.append(num)
            elif group:
                yield group
                group = []
        if group:
            yield group

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
            "mc_token_ids": []
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates
            if self.args.eval_all_snippets:
                candidate_keys = list(self.snippets.keys())
            else:
                candidate_keys = example["candidates"]
            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]
        else:
            if self.args.selection_type == "all":
                if self.args.negative_sample_method == "all":
                    candidate_keys = list(self.snippets.keys())
                elif self.args.negative_sample_method == "mix":
                    candidate_keys = example["candidates"] + random.sample(list(self.snippets.keys()), k=len(example["candidates"]))
                elif self.args.negative_sample_method == "oracle":
                    candidate_keys = example["candidates"]
                else: # although we have already checked for this, still adding this here to be sure
                    raise ValueError("negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)
                candidates = [self.snippets[cand_key] for cand_key in candidate_keys]
        if self.args.selection_type == "domain":
            candidate_keys = list(self.domains.keys())
            candidates = list(self.domains.values())
        elif self.args.selection_type == "domain_entity":
            candidates = [v for domain in self.domains.keys() for v in self.entities[domain].values()]
            candidate_keys = [f"{domain}__{entity_id}" for domain in self.domains.keys() for entity_id in self.entities[domain].keys()]
            if len(candidates) < self.args.n_candidates:
                # If the domain has not enough entities, add examples from other domains
                candidates = list(chain(*[list(v.values()) for v in self.entities.values()]))
                candidate_keys = [f"{candidate_domain}__{entity_id}" for candidate_domain in list(self.domains.keys()) for entity_id in self.entities[candidate_domain].keys()]
        elif self.args.selection_type == "entity":
            domain = example['label']["knowledge"][0]['domain']
            candidates = list(self.entities[domain].values())
            candidate_keys = [f"{domain}__{entity_id}" for entity_id in self.entities[domain].keys()]
            if len(candidates) < self.args.n_candidates:
                # If the domain has not enough entities, add examples from other domains
                candidates = list(chain(*[list(v.values()) for v in self.entities.values()]))
                candidate_keys = [f"{candidate_domain}__{entity_id}" for candidate_domain in list(self.domains.keys()) for entity_id in self.entities[candidate_domain].keys()]
        elif self.args.selection_type == "doc":
            domain = example['label']["knowledge"][0]['domain']
            entity_id = example['label']["knowledge"][0]['entity_id']
            prefix = f"{domain}__{entity_id}__"
            general_prefix = f"{domain}__*__"
            candidate_keys = [cand for cand in self.snippets.keys() if cand.startswith(prefix) or cand.startswith(general_prefix)]
            candidates = [self.snippets[cand_key] for cand_key in candidate_keys]

        this_inst["candidate_keys"] = candidate_keys

        if self.split_type == "train":
            # Sample args.n_candidates from candidates
            candidates = self._shrink_label_cands(example["knowledge"], candidates)

        label_idx = candidates.index(example["knowledge"])
            
        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance, _ = self.build_input_from_segments(
                cand,
                example["history"],
                example["current_turn"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
            this_inst["mc_token_ids"].append(instance["mc_token_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history, current_turn):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}

        if self.args.multitask:
            sequence = [[self.bos]] + [knowledge + [self.knowledge_tag]] + history + [current_turn] + [[self.eos]]
        else:
            sequence = [[self.bos]] + history + [current_turn] + [[self.knowledge_tag] + knowledge] + [[self.eos]]

        if self.args.type_vocab_size == self.args.vocab_size:
            token_type_ids = [[s[0]] * len(s) for i, s in enumerate(sequence)]
            token_type_ids = list(chain(*token_type_ids))
        # BERT and alike models
        elif self.args.type_vocab_size == 2:
            if self.args.multitask:
                token_type_ids = [0] +  [1] * (len(sequence[1])) + [0 for _, s in enumerate(sequence[2:]) for _ in s]
            else:
                token_type_ids = [0 for _, s in enumerate(sequence[:-2]) for _ in s] + [0] + [1] * (len(sequence[-2]))
        else:
            token_type_ids = [0 for _, s in enumerate(sequence) for _ in s]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = token_type_ids
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence
    
    def _shrink_label_cands(self, label, candidates):
        shrunk_label_cands = candidates.copy()
        shrunk_label_cands.remove(label)
        shrunk_label_cands = random.sample(shrunk_label_cands, k=self.args.n_candidates-1)
        shrunk_label_cands.append(label)
        random.shuffle(shrunk_label_cands)

        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        label_idx = [ins["label_idx"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        batch_size = len(batch)
        n_candidates = len(batch[0]["input_ids"])
        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        ).view(batch_size, n_candidates, -1)
        
        token_type_pad = token_type_ids[0][-1] if self.args.type_vocab_size != self.args.vocab_size else self.pad
        token_type_ids = torch.tensor(
            pad_ids(token_type_ids, token_type_pad)
        ).view(batch_size, n_candidates, -1)

        lm_labels = torch.full_like(input_ids, -100)
        mc_token_ids = torch.tensor(mc_token_ids).view(batch_size, n_candidates)
        label_idx = torch.tensor(label_idx)

        return input_ids, token_type_ids, mc_token_ids, lm_labels, label_idx, data_info


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, history, current_turn):
        """ Build a sequence of input from history """
        instance = {}

        if self.args.multitask:
            current_turn_sep = []
        else:
            current_turn_sep = [self.knowledge_tag]
        sequence = [[self.bos]] + history + [current_turn_sep + current_turn] + [[self.eos]]

        if self.args.type_vocab_size == self.args.vocab_size:
            token_type_ids = [[s[0]] * len(s) for i, s in enumerate(sequence)]
            token_type_ids = list(chain(*token_type_ids))
        elif self.args.type_vocab_size == 2:
            token_type_ids = [0 for _, s in enumerate(sequence[:-2]) for _ in s] + [0] * len(current_turn_sep) + [1] * (len(sequence[-2]) - len(current_turn_sep)) + [1]
        else:
            token_type_ids = [0 for _, s in enumerate(sequence) for _ in s]

        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = token_type_ids
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["history"],
            example["current_turn"]
        )
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        mc_token_ids = [ins["mc_token_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_pad = token_type_ids[0][-1] if self.args.type_vocab_size != self.args.vocab_size else self.pad
        token_type_ids = torch.tensor(pad_ids(token_type_ids, token_type_pad))
        mc_token_ids = torch.tensor(mc_token_ids)
        lm_labels = torch.full_like(input_ids, -100)
        labels = torch.tensor(labels).float()

        return input_ids, token_type_ids, mc_token_ids, lm_labels, labels, data_info


class MultiTaskDataset(BaseDataset):
    """
    This class contains both the Dataset for tnowledge-seeking turn detection
    and knowledge selection
    """
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        args.multitask = True
        self.datasets = {
            "detection": {
                "data" :KnowledgeTurnDetectionDataset(self._align_args(args, "detection"), tokenizer, split_type, labels=labels, labels_file=labels_file),
            },
            "selection": {
                "data": KnowledgeSelectionDataset(self._align_args(args, "selection"), tokenizer, split_type, labels=labels, labels_file=labels_file),
            },
            "generation": {
                "data": ResponseGenerationDataset(self._align_args(args, "generation"), tokenizer, split_type, labels_file=labels_file),
            }
        }

        end = -1
        for task, dataset in self.datasets.items():
            start = end + 1
            end = start + len(dataset["data"]) - 1
            self.datasets[task]["range"] = (start, end)
            
        args.task = "multitask"

    def _align_args(self, args, task):
        args.task = task
        return args

    def __getitem__(self, index):
        """Returns a dict of the relevant (task-dependent) information for the item."""
        for task, dataset in self.datasets.items():
            if dataset["range"][0] <= index <= dataset["range"][1]:
                instance = dataset["data"][index % len(dataset["data"])]
                instance["task"] = task
                return instance

        raise Exception("Index out of bounds")

    def collate_fn(self, batch):
        """
        Splits the batch into mini-batches corresponding to the sample task.

        Returns:
            A dict with keys detection, selection and generation which contain a
            list of collated instances corresponding to the task.
        """
        data = {
            "detection": [],
            "selection": [],
            "generation": []
        }

        for instance in batch:
            task = instance["task"]
            data[task].append(instance)

        for task, mini_batch in data.items():
            if len(mini_batch) > 0:
                data[task] = self.datasets[task]["data"].collate_fn(mini_batch)

        return data

    def __len__(self):
        return sum([len(dataset["data"]) for _, dataset in self.datasets.items()])


class EmbeddingDataset(KnowledgeSelectionDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(EmbeddingDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"]
        }

        total_num_examples = len(self.examples)
        total_num_snippets = len(self.snippets)

        negative_sample_snippets = random.random() <= total_num_snippets / (total_num_examples + total_num_snippets)

        positive_example = example["history"] + [example["current_turn"]]
        positive_snippet = [example['knowledge']]

        n_sample = 10
        # TODO consider mining of hard examples
        if negative_sample_snippets:
            snippet_candidates = random.sample(list(self.snippets.values()), k=n_sample)
            positive_index = snippet_candidates.index(positive_snippet) if positive_snippet in snippet_candidates else -1
            negative_samples = [[snippet_candidates[index]] for index in set(range(n_sample)) if index != positive_index]
        else:
            negative_sample_index = random.choices(list(chain(range(index), range(index + 1, total_num_examples))), k=n_sample)
            negative_examples = [self.examples[index] for index in negative_sample_index]
            negative_samples = [negative_example["history"] + [negative_example["current_turn"]] for negative_example in negative_examples]
        
        anchor = positive_example if negative_sample_snippets else positive_snippet
        positive_sample = positive_snippet if negative_sample_snippets else positive_example

        anchor_ids, positive_ids, negative_ids = self.build_input_from_segments(
            anchor, positive_sample, negative_samples
        )

        this_inst['anchor_ids'] = anchor_ids
        this_inst['positive_ids'] = positive_ids
        this_inst['negative_ids'] = negative_ids
        
        this_inst["negative_sample_snippets"] = negative_sample_snippets

        return this_inst

    def build_input_from_segments(self, anchor, positive, negatives):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        
        anchor_sequence = [[self.bos]] + anchor + [[self.eos]]
        positive_sequence = [[self.bos]] + positive + [[self.eos]]
        negative_sequences = [[[self.bos]] + negative + [[self.eos]] for negative in negatives]
        # instance["input_ids"] = list(chain(*sequence))
        # instance["token_type_ids"] = token_type_ids
        # instance["mc_token_ids"] = len(instance["input_ids"]) - 1

        return list(chain(*anchor_sequence)), list(chain(*positive_sequence)), [list(chain(*negative_sequence)) for negative_sequence in negative_sequences]

    def collate_fn(self, batch):
        anchor_ids = [ins["anchor_ids"] for ins in batch]
        positive_ids = [ins["positive_ids"] for ins in batch]
        negative_ids = list(chain(*[ins["negative_ids"] for ins in batch]))
        # token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        # mc_token_ids = [id for ins in batch for id in ins["mc_token_ids"]]
        # label_idx = [ins["label_idx"] for ins in batch]
        #negative_ids = list(chain(*negative_ids))

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "negative_sample_snippets": [ins["negative_sample_snippets"] for ins in batch]
        }

        batch_size = len(batch)
        # input_ids = anchor_ids + positive_ids + negative_ids
        # input_ids = torch.tensor(
        #     pad_ids(input_ids, self.pad)
        # )
        input_ids = [torch.tensor(
            pad_ids(anchor_ids, self.pad)
        ), torch.tensor(
            pad_ids(positive_ids, self.pad)
        ), torch.tensor(
            pad_ids(negative_ids, self.pad)
        )]

        data_info["batch_size"] = batch_size

        return input_ids, data_info


class KnowledgeEmbeddingDataset(EmbeddingDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeEmbeddingDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __len__(self):
        return len(self.knowledge_docs)

    def __getitem__(self, index):
        knowledge = self.knowledge_docs[index]
        key = "{}__{}__{}".format(knowledge["domain"], str(knowledge["entity_id"]) or "*", knowledge["doc_id"])

        this_inst = {
            "domain": knowledge['domain'],
            "entity_id": knowledge['entity_id'],
            "doc_id": knowledge['doc_id'],
            "key": key
        }

        snippet = self.snippets[key]
        input_ids = self.build_input_from_segments([snippet])
        this_inst['input_ids'] = input_ids

        return this_inst

    def build_input_from_segments(self, input):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        
        sequence = [[self.bos]] + input + [[self.eos]]

        return list(chain(*sequence))

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]

        data_info = {
            "domains": [ins["domain"] for ins in batch],
            "entity_ids": [ins["entity_id"] for ins in batch],
            "doc_ids": [ins["doc_id"] for ins in batch],
            "key": [ins["key"] for ins in batch]
        }

        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        )

        return input_ids, data_info


class DialogEmbeddingDataset(EmbeddingDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(DialogEmbeddingDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "knowledge_key": example["knowledge_key"]
        }

        this_inst['candidate_keys'] = list(self.snippets.keys())

        input = example["history"] + [example["current_turn"]]
        input_ids = self.build_input_from_segments(input)
        this_inst['input_ids'] = input_ids

        return this_inst
        
    def build_input_from_segments(self, input):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        
        sequence = [[self.bos]] + input + [[self.eos]]

        return list(chain(*sequence))

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]

        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "knowledge_key": [ins["knowledge_key"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        input_ids = torch.tensor(
            pad_ids(input_ids, self.pad)
        )

        return input_ids, data_info

class JointSelectionDataset:

    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        args.multitask = True
        args.task = "multitask"
        self.datasets = {}
        for selection_type in args.train_joint_selection_types:
            self.datasets[selection_type] = {}
            self.datasets[selection_type]["data"] = KnowledgeSelectionDataset(self._align_args(args, selection_type), tokenizer, split_type, labels=labels, labels_file=labels_file)
            
        end = -1
        for task, dataset in self.datasets.items():
            start = end + 1
            end = start + len(dataset["data"]) - 1
            self.datasets[task]["range"] = (start, end)

    def _align_args(self, args, selection_type):
        args.selection_type = selection_type
        return copy.deepcopy(args)

    def __getitem__(self, index):
        """Returns a dict of the relevant (task-dependent) information for the item."""
        items = {}
        
        for selection_type, dataset in self.datasets.items():
            items[selection_type] = dataset[index]

        return items

    def collate_fn(self, batch):
        """Splits the batch into mini-batches corresponding to the selection task"""
        data = {
            k: [] for k in self.args.train_joint_selection_types
        }

        for instance in batch:
            for selection_type, item in instance.items():
                data[selection_type].append(item)

        for selection_type, mini_batch in data.items():
            if len(mini_batch) > 0:
                data[selection_type] = self.datasets[selection_type].collate_fn(mini_batch)

        return data

    def __len__(self):
        return len(self.datasets["domain"])
