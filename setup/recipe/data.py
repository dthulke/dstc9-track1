import json

from sisyphus import *


class MakeTrueTargets(Job):
    def __init__(self, orig):
        self.orig = orig
        self.labels = self.output_path('labels.json')

    def run(self):
        with open(self.orig) as in_fp, open(self.labels, 'w+') as out_fp:
            labels = json.load(in_fp)
            def map_target_to_true(sample):
                sample['target'] = True
                if 'knowledge' in sample:
                    del sample['knowledge']
                if 'response' in sample:
                    del sample['response']
                return sample
            out_labels = [map_target_to_true(l) for l in labels]
            json.dump(out_labels, out_fp)

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)

class CreateLabelDifference(Job):

    def __init__(self, old, new):
        self.old = old
        self.new = new
        self.labels = self.output_path('labels.json')

    def run(self):
        with open(self.old, "r") as f_old, open(self.new, "r") as f_new, open(self.labels, 'w+') as f_out:
            old = json.load(f_old)
            new = json.load(f_new)
            
            def map_labels_for_recalculation(old, new):
                # recalculation only needs to be done for those that are newly identified as
                # knowledge-seeking turns
                new['target'] = new['target'] and not old['target']
                if 'knowledge' in new:
                    del new['knowledge']
                if 'response' in new:
                    del new['response']
                return new

            out_labels = [map_labels_for_recalculation(old_label, new_label) for old_label, new_label in zip(old, new)]
            json.dump(out_labels, f_out)

        def tasks(self):
            yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)

class MergeSelectionLabels(Job):

    def __init__(self, new_detection, old_selection, new_selection):
        self.new_detection = new_detection
        self.old_selection = old_selection
        self.new_selection = new_selection
        self.labels = self.output_path('labels.json')

    def run(self):
        with open(self.new_detection, "r") as f_new_detection, \
         open(self.old_selection, "r") as f_old_selection,  \
         open(self.new_selection, "r") as f_new_selection, \
         open(self.labels, 'w+') as f_out:
            new_detection = json.load(f_new_detection)
            old_selection = json.load(f_old_selection)
            new_selection = json.load(f_new_selection)
            
            def get_label(old_label, new_label, new_detection_label):
                if new_label["target"]:
                    # new positives
                    return new_label
                elif old_label["target"] and new_detection_label["target"]:
                    # positive in old and new run, thus not recalculated
                    return old_label
                else:
                    # otherwise negative in new run
                    return {"target": False}

            labels = []
            for old_label, new_label, new_detection_label in zip(old_selection, new_selection, new_detection):
                labels.append(get_label(old_label, new_label, new_detection_label))
                        
            json.dump(labels, f_out)

        def tasks(self):
            yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)


class CreateOutOfDomainTestData(Job):
    __sis_hash_exclude__ = {
        'remove_ood_knowledge': False
    }

    def __init__(self, dataset, train_domain_pred, validation_domain_pred, domains, remove_ood_knowledge=False):
        self.dataset = dataset
        self.domain_preds = {
            'train': train_domain_pred,
            'val': validation_domain_pred
        }
        self.domains = domains
        self.remove_ood_knowledge = remove_ood_knowledge

        # Creates a new data directory
        ## train - contains the in domain training data
        ## val - contains the in domain validation data
        ## test - contains the out of domain validation data
        self.data = self.output_path('data', directory=True)
        self.test_data = self.output_path('test_data', directory=True)

    def run(self):
        import os
        import json
        import shutil

        for type, domain_pred in self.domain_preds.items():
            new_logs = []
            new_labels = []
            if type == 'val':
                new_test_logs = []
                new_test_labels = []
            
            with open(self.dataset + f'/{type}/logs.json') as logs_fp, \
                    open(self.dataset + f'/{type}/labels.json') as labels_fp, \
                    open(domain_pred) as preds_fp:
                logs = json.load(logs_fp)
                labels = json.load(labels_fp)
                preds = json.load(preds_fp)
                for log, label, pred in zip(logs, labels, preds):
                    domain = label['knowledge'][0]['domain'] \
                        if label['target'] else pred['knowledge'][0]['domain']
                    if domain in self.domains:
                        new_logs.append(log)
                        new_labels.append(label)
                    elif type == 'val':
                        new_test_logs.append(log)
                        new_test_labels.append(label)

            type_path = os.path.join(self.data.get_path(), type)
            if not os.path.exists(type_path):
                os.makedirs(type_path)

            with open(os.path.join(type_path, 'logs.json'), 'w+') as fp:
                json.dump(new_logs, fp)
            with open(os.path.join(type_path, 'labels.json'), 'w+') as fp:
                json.dump(new_labels, fp)

            if type == 'val':
                type_path = os.path.join(self.test_data.get_path(), 'test')
                if not os.path.exists(type_path):
                    os.makedirs(type_path)

                with open(os.path.join(type_path, 'logs.json'), 'w+') as fp:
                    json.dump(new_test_logs, fp)
                with open(os.path.join(type_path, 'labels.json'), 'w+') as fp:
                    json.dump(new_test_labels, fp)

            # Copy knowledge json
            source_knowledge_path = self.dataset + '/knowledge.json'
            if not self.remove_ood_knowledge:
                shutil.copyfile(source_knowledge_path, self.data.get_path() + '/knowledge.json')
                shutil.copyfile(source_knowledge_path, self.test_data.get_path() + '/knowledge.json')
            else:
                with open(source_knowledge_path) as fp:
                    source_knowledge = json.load(fp)
                    with open(self.data.get_path() + '/knowledge.json', 'w') as in_fp:
                        in_domain_knowledge = {k: v for k, v in source_knowledge.items() if k in self.domains}
                        json.dump(in_domain_knowledge, in_fp)
                shutil.copyfile(source_knowledge_path, self.test_data.get_path() + '/knowledge.json')

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)


class ConcatLabels(Job):
    def __init__(self, label1_path, label2_path):
        self.label1_path = label1_path
        self.label2_path = label2_path
        self.labels = self.output_path('labels.json')

    def run(self):
        import json
        labels1 = json.load(open(self.label1_path))
        labels2 = json.load(open(self.label2_path))

        with open(self.labels, 'w') as out_fp:
            json.dump(labels1 + labels2, out_fp)

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
