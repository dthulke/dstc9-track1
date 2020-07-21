import collections
import json
from functools import partial

from dataset_walker import DatasetWalker

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

if __name__ == "__main__":
    pred_file_path = "/work/smt2/daheim/dstc9_baseline/baseline_val.json"
    val_dataroot = "/u/daheim/alexa-with-dstc9-track1-dataset/data/"

    swaps = filter_swapped_domains(pred_file_path, val_dataroot)
    with open("/u/daheim/alexa-with-dstc9-track1-dataset/swaps.json", "w") as f:
        json.dump(swaps, f)

    swaps = filter_swapped_entities(pred_file_path, val_dataroot)
    with open("/u/daheim/alexa-with-dstc9-track1-dataset/entity_swaps.json", "w") as f:
        json.dump(swaps, f)