import json
from scripts.knowledge_reader import KnowledgeReader

def print_context(log):
    for utterance in log:
        print(f"  {utterance['speaker']}: {utterance['text']}")

def analyze_errors(logs_file, knowledge_file, ref_file, pred_file):
    with open(logs_file) as logs_fp, open(ref_file) as ref_fp, open(pred_file) as pred_fp:
        logs = json.load(logs_fp)
        knowledge = KnowledgeReader(None, knowledge_file)
        ref_samples = json.load(ref_fp)
        pred_samples = json.load(pred_fp)

        for log, ref_sample, pred_sample in zip(logs, ref_samples, pred_samples):
            if ref_sample['target'] != pred_sample['target']:
                print(f"Target wrong. Pred {pred_sample['target']} and ref {ref_sample['target']}")
                print_context(log)
            elif ref_sample['target'] and ref_sample['knowledge'][0] != pred_sample['knowledge'][0]:
                print(f"Knowledge wrong. Pred {pred_sample['knowledge'][0]} and ref {ref_sample['knowledge'][0]}")
                print_context(log)
                print(f"    Target doc: {knowledge.get_doc(**ref_sample['knowledge'][0])}")
                print(f"    Pred doc:")
                for doc_id in pred_sample['knowledge']:
                    print(f"      {knowledge.get_doc(**doc_id)}")
                print(f"  Ref response: {ref_sample['response']}")
                print(f"  Pred response: {pred_sample['response']}")
            elif ref_sample['target']:
                print(f"Correct")
                print_context(log)
                print(f"    Doc: {knowledge.get_doc(**ref_sample['knowledge'][0])}")
                print(f"  Ref response: {ref_sample['response']}")
                print(f"  Pred response: {pred_sample['response']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("logs_file", type=str)
    parser.add_argument("knowledge_file", type=str)
    parser.add_argument("ref_file", type=str)
    parser.add_argument("pred_file", type=str)

    args = parser.parse_args()

    analyze_errors(args.logs_file, args.knowledge_file, args.ref_file, args.pred_file)