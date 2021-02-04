import collections
import json
import re
from functools import partial

import matplotlib.pyplot as plt

from dataset_walker import DatasetWalker
from knowledge_reader import KnowledgeReader

class Dataset(object):

    def __init__(self, examples):
        self.examples = examples
    
    def __getitem__(self, index):
        example = self.examples[index]
        return example

def filter_swapped_domains(pred_file_path, val_dataroot):
    with open(pred_file_path, 'r') as f:
        preds = json.load(f)
    eval_dataset = DatasetWalker("val", val_dataroot, labels=True)

    total = collections.defaultdict(int)
    swaps = collections.defaultdict(partial(collections.defaultdict, float))
    for (_, ref), pred in zip(eval_dataset, preds):
        # only consider tp knowledge-seeking turn detections
        if ref["target"] and pred["target"]:
            ref_domain = ref["knowledge"][0]["domain"]
            pred_domain = pred["knowledge"][0]["domain"]
            total[ref_domain] += 1
            if ref_domain != pred_domain:
                swaps[ref_domain][pred_domain] += 1
    for ref_domain, false_domains in swaps.items():
        for domain, no_swaps in false_domains.items():
            swaps[ref_domain][domain] = no_swaps / total[ref_domain]
    return swaps

def filter_swapped_entities(pred_file_path, val_dataroot):
    with open(pred_file_path, 'r') as f:
        preds = json.load(f)
    eval_dataset = DatasetWalker("val", val_dataroot, labels=True)

    total = collections.defaultdict(int)
    swaps = collections.defaultdict(float)
    for (_, ref), pred in zip(eval_dataset, preds):
        # only consider tp knowledge-seeking turn detections
        if ref["target"] and pred["target"]:
            ref_domain = ref["knowledge"][0]["domain"]
            pred_domain = pred["knowledge"][0]["domain"]
            total[ref_domain] += 1
            if ref_domain == pred_domain:
                ref_entity = ref["knowledge"][0]["entity_id"]
                pred_entity = pred["knowledge"][0]["entity_id"]
                if ref_entity != pred_entity:
                    swaps[ref_domain] += 1

    for ref_domain, no_swaps in swaps.items():
        swaps[ref_domain] = no_swaps / total[ref_domain]

    return swaps

def filter_swapped_documents(pred_file_path, val_dataroot):
    with open(pred_file_path, 'r') as f:
        preds = json.load(f)
    eval_dataset = DatasetWalker("val", val_dataroot, labels=True)

    total = collections.defaultdict(int)
    swaps = collections.defaultdict(float)
    for (log, ref), pred in zip(eval_dataset, preds):
        # only consider tp knowledge-seeking turn detections
        if ref["target"] and pred["target"]:
            ref_domain = ref["knowledge"][0]["domain"]
            pred_domain = pred["knowledge"][0]["domain"]
            if ref_domain == pred_domain:
                ref_entity = ref["knowledge"][0]["entity_id"]
                pred_entity = pred["knowledge"][0]["entity_id"]
                if ref_entity == pred_entity:
                    total[ref_domain] += 1
                    ref_doc = ref["knowledge"][0]["doc_id"]
                    pred_doc = pred["knowledge"][0]["doc_id"]
                    if ref_doc != pred_doc:
                        swaps[ref_domain] += 1

    for ref_domain, no_swaps in swaps.items():
        swaps[ref_domain] = no_swaps / total[ref_domain]

    return swaps

def plot_entity_mentions(val_dataroot, knowledge_file):
    re_punc = re.compile(r'[!,.?]')
    eval_dataset = DatasetWalker("val", val_dataroot, labels=True)
    knowledge_reader = KnowledgeReader(val_dataroot, knowledge_file)
    knowledge = knowledge_reader.knowledge
    knowledge_docs = knowledge_reader.get_doc_list()
    lens = []

    for log, labels in eval_dataset:
        if labels["target"]:
            domain = labels["knowledge"][0]["domain"]
            entity_id = labels["knowledge"][0]["entity_id"]
            entity_name = knowledge_reader.get_entity_name(domain, entity_id)
            if entity_name is None:
                continue
            entity_name = entity_name.lower()
            history = [turn["text"].lower() for turn in log]
            history = " ".join([re_punc.sub(" ", text) for text in history])
            occurences = [occurence.start() for occurence in re.finditer(entity_name, history)]
            lens.extend([len(history) - occurence for occurence in occurences])

    plt.hist(lens, density=True, bins=25)
    for idx in [128, 256, 512, 1024]:
        plt.axvline(idx, color="purple", linewidth=1)
        plt.text(idx+10, .0025, str(idx), rotation=0)
    plt.title("No. of history tokens needed for entity to be considered")
    plt.xlabel("No. of needed history tokens")
    plt.savefig("entity_history.png")


if __name__ == "__main__":
    pred_file_path_128 = "/work/smt2/daheim/dstc9_baseline/baseline_val.json"
    pred_file_path_512 = "/u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/RunBaselineJob.DuIwE7VkUaAq/output/preds.json"
    val_dataroot = "/u/daheim/alexa-with-dstc9-track1-dataset/baseline/data/"
    knowledge_file = val_dataroot + "knowledge.json"

    swaps_128 = filter_swapped_entities(pred_file_path_128, val_dataroot)
    swaps_512 = filter_swapped_entities(pred_file_path_512, val_dataroot)
    print("Change in swapped entities within domain: ")
    for key, value in swaps_128.items():
        print("{} ".format(key))
        new_value = swaps_512.get(key, 0)
        print("{}: {} -> {}".format(
            key,
            "{:.2f}%".format(value * 100), 
            "{:.2f}%".format(new_value * 100)))
    print("\n")

    swaps_128 = filter_swapped_domains(pred_file_path_128, val_dataroot)
    swaps_512 = filter_swapped_domains(pred_file_path_512, val_dataroot)
    #change = {}
    #for key, dict_ in swaps_128.items():
    #    change[key] = {key_: dict_[key_] - swaps_512.get(key, 0).get(key_, 0) for key_ in dict_}
    print("Change in swapped domains: ")
    for key, dict_ in swaps_128.items():
        print("Swaps between {} and ".format(key))
        for key_, value in dict_.items():
            new_value = swaps_512.get(key, 0).get(key_, 0)
            print("\t{}: {} -> {}".format(
                key_, 
                "{:.2f}%".format(value * 100),
                "{:.2f}%".format(new_value * 100)))
    print("\n")
    
    swaps_128 = filter_swapped_documents(pred_file_path_128, val_dataroot)
    swaps_512 = filter_swapped_documents(pred_file_path_512, val_dataroot)
    change = {key: swaps_128[key] - swaps_512.get(key, 0) for key in swaps_128}
    print("Change in swapped documents within domain: ")
    for key, value in swaps_128.items():
        print("Swaps within {} ".format(key))
        new_value = swaps_512.get(key, 0)
        print("{}: {} -> {}".format(
            key, 
            "{:.2f}%".format(value * 100),
            "{:.2f}%".format(new_value * 100)))
    #plot_entity_mentions(val_dataroot, knowledge_file)
