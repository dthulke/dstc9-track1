import collections
import json
from functools import partial

import numpy as np

from dataset_walker import DatasetWalker
from knowledge_reader import KnowledgeReader
from scores import Metric


def get_scores(ref, hyp):
    """
    Calculates all NLG evaluation metrics and returns the score in a dict.

    Args:
        ref: reference string.
        hyp: string to compare to the reference.

    Returns:
        Dict indexed by a key indicating the used metric and the corresponding score.
    """
    scores = {}
    metric = Metric()
    bleu_modes = range(1, 5)
    rogue_modes = [1, 2, "l"]

    for bleu_mode in bleu_modes:
        scores["bleu_"+str(bleu_mode)] = metric._bleu(ref, hyp, bleu_mode)

    for rogue_mode in rogue_modes:
        if len(hyp.strip()) == 0 or len(ref.strip()) == 0:
            scores["rogue_"+str(rogue_mode)] = 0.0
        else:
            scores["rogue_"+str(rogue_mode)] = metric._rouge(ref, hyp, rogue_mode)

    scores["meteor"] = metric._meteor(ref, hyp)

    return scores

def highest_changes(bases, comps, logs, labels, knowledge, n=50):
    """
    Retrieves information about the n most differing responses generated in
    base and comparison datasets.

    Args:
        bases: file of responses and their information to compare against.
        comps: file of responses and their information to compare.
        logs: dict of chatlogs.
        labels: list of tuples containing the ground truth labels.
        n: Number of responses to return.

    Returns:
        dict containing the calculated NLG evaluation metric, the responses,
        selected knowledge and ground truth information.
    """
    with open(bases, 'r') as f:
        bases = json.load(f)
    
    with open(comps, 'r') as f:
        comps = json.load(f)

    scores = collections.defaultdict(partial(collections.defaultdict, float))
    for i, (base, comp, (_, label)) in enumerate(zip(bases, comps, labels)):
        if base["target"] and comp["target"]:
            scores[str(i)]["scores"] = get_scores(base["response"], comp["response"])
            scores[str(i)]["base"] = {
                "response": base["response"],
                "knowledge": knowledge.get_doc(
                    base["knowledge"][0]["domain"], 
                    base["knowledge"][0]["entity_id"], 
                    base["knowledge"][0]["doc_id"]
                )
            }
            scores[str(i)]["compared"] = {
                "response": comp["response"],
                "knowledge": knowledge.get_doc(
                    comp["knowledge"][0]["domain"], 
                    comp["knowledge"][0]["entity_id"], 
                    comp["knowledge"][0]["doc_id"]
                )
            }
            if label["target"]:
                scores[str(i)]["ground_truth"] = {
                    "response": label["response"],
                    "knowledge": knowledge.get_doc(
                        label["knowledge"][0]["domain"], 
                        label["knowledge"][0]["entity_id"], 
                        label["knowledge"][0]["doc_id"]
                    )
                }

    def sort_fn(item):
        return np.mean([score for _, score in item[1]["scores"].items()])

    sorted_ = {k: scores_ for k, scores_ in 
               sorted(scores.items(), key=sort_fn, reverse=False)[:50]}

    for i, _ in sorted_.items():
         log = logs[int(i)]
         log = [d["text"] for d in log]
         sorted_[i]["log"] = log

    return sorted_
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--ref", type=str)
    parser.add_argument("--hyp", type=str)

    args = parser.parse_args()
    #args.ref_file = "/u/thulke/work/setups/sisyphus/dstc9-track1/work/baseline/RunBaselineJob.1c5RdTsoafHP/output/preds.json"
    #args.hyp_file = "/u/thulke/work/setups/sisyphus/dstc9-track1/work/baseline/RunBaselineJob.7XYYauKnAm6Y/output/preds.json"
    args.dataroot = "/u/daheim/alexa-with-dstc9-track1-dataset/baseline/data/"
    args.dataset = "val"

    labels = DatasetWalker(dataroot=args.dataroot, dataset=args.dataset, labels=True)
    knowledge = KnowledgeReader(args.dataroot, 'knowledge.json')

    logs_file = args.dataroot + args.dataset + "/logs.json"

    with open(logs_file, 'r') as f:
        logs = json.load(f)

    changes = highest_changes(args.ref, args.hyp, logs, labels, knowledge)

    with open("changes.json", "w") as f:
        json.dump(changes, f, indent=4, separators=(',', ': '))