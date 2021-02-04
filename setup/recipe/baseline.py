import os
import gzip
import glob
import sisyphus.hash
import time
import math
import pathlib
import json

from sisyphus import *


class TrainBaselineJob(Job):
    __sis_hash_exclude__ = {
        'extra_time': 0,
        'extra_memory': 0,
        'data_path': None,
        'distributed': True,
    }

    def __init__(self, name, config, additional_args="", gpus=2, data_path=None, extra_time=0, extra_memory=0, distributed=True):
        self.name = name
        self.additional_args = additional_args
        self.gpus = gpus
        
        self.extra_time = extra_time
        self.extra_memory = extra_memory

        if data_path is None:
            baseline_path = pathlib.Path(gs.BASELINE_ROOT)
            self.data_path = baseline_path / 'data'
        else:
            self.data_path = data_path

        self.config = config
        self.distributed = distributed

        self.checkpoint = self.output_path('checkpoint', directory=True)

    def write_config(self):
        def dumper(obj):
            if isinstance(obj, Path):
                return obj.get_cached_path()
            if 'toJSON' in obj.__attrs__():
                return obj.toJSON()
            else:
                return str(obj)

        with open('params.json', 'w') as fp:
            json.dump(self.config, fp, default=dumper)

    def run(self):
        import os
        import pathlib

        runs_path = pathlib.Path('runs')
        runs_path.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.checkpoint) or not (os.path.islink(self.checkpoint) or os.path.islink(runs_path / self.name)):
            os.symlink(src=self.checkpoint,
                    dst=runs_path / self.name,
                    target_is_directory=True)

        distributed_part = ""
        if self.gpus > 1 and self.distributed:
            # Find a free port which can be used for distributed training
            import socket
            free_port = None
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for port in range(9000, 10000):
                try:
                    sock.bind(('', port))
                    sock.close()
                    free_port = port
                    break
                except OSError:
                    # Try next port
                    pass

            if free_port is None:
                raise Exception("No free port for distributed training available.")
            
            distributed_part =  f"-m torch.distributed.launch --nproc_per_node {self.gpus} --master_port {free_port}"

        self.sh(
            '{python} {distributed_part} {baseline_path}/baseline.py --dataroot {dataroot} --exp_name {name} --params_file params.json {additional_args}',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            distributed_part=distributed_part,
            dataroot=self.data_path,
            additional_args=self.additional_args,
            name=self.name,
        )

    def tasks(self):
        yield Task('write_config', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
        yield Task('run', resume='run', rqmt={
            'cpu': self.gpus,
            'gpu': self.gpus,
            'mem': 14 * self.gpus + self.extra_memory,
            'time': 24 / self.gpus + 4 + self.extra_time,
            'qsub_args': "-l h='*1080*'"
        })


class RunBaselineJob(Job):
    __sis_hash_exclude__ = {
        'config': None,
        'extra_time': 0,
        'data_path': None,
        'eval_split': 'val',
        'split': 1
    }

    def __init__(self, checkpoint, config=None, labels=None, additional_args="", data_path=None, eval_split='val', gpus=1, extra_time=0, split=1):
        self.checkpoint = checkpoint

        self.labels = labels
        self.additional_args = additional_args
        self.config = config

        if data_path is None:
            self.data_path = pathlib.Path(gs.BASELINE_ROOT) / 'data'
        else:
            self.data_path = data_path
        self.eval_split = eval_split
        
        self.preds = self.output_path('preds.json')
        self.gpus = gpus
        self.extra_time = extra_time
        
        self.split = split
        self.new_data_path = 'split_data'

    def split_labels(self):
        import os
        import json
        import shutil

        knowledge_path = os.path.join(self.data_path, 'knowledge.json')
        log_path = os.path.join(self.data_path, self.eval_split, 'logs.json')
        label_path = self.labels

        with open(label_path) as fp:
            labels = json.load(fp)
        with open(log_path) as fp:
            logs = json.load(fp)
        
        assert len(logs) == len(labels)
        num_labels = len(labels)

        os.mkdir(self.new_data_path)

        for split_id in range(self.split):
            new_data_path = os.path.join(self.new_data_path, str(split_id))
            os.mkdir(new_data_path)

            shutil.copyfile(knowledge_path, os.path.join(new_data_path, 'knowledge.json'))

            new_data_path = os.path.join(new_data_path, self.eval_split)
            os.mkdir(new_data_path)

            low = (num_labels // self.split) * split_id
            upper = (num_labels // self.split) * (split_id + 1) if split_id != (self.split - 1) else num_labels
            
            with open(os.path.join(new_data_path, 'logs.json'), 'w') as fp:
                json.dump(logs[low:upper], fp)

            with open(os.path.join(new_data_path, 'labels.json'), 'w') as fp:
                json.dump(labels[low:upper], fp)

    def merge_preds(self):
        import json
        preds = []
        for split_id in range(self.split):
            with open(f'preds.{split_id}.json') as fp:
                preds += json.load(fp)
        with open(self.preds, 'w') as fp:
            json.dump(preds, fp)

    def run(self, split_id=-1):
        if self.config is not None:
            with open('params.json', 'w') as fp:
                json.dump(self.config, fp)

        data_path = self.data_path if split_id == -1 else os.path.join(self.new_data_path, str(split_id))        
        labels = self.labels if split_id == -1 else os.path.join(data_path, self.eval_split, 'labels.json')

        if self.labels is None:
            label_part = "--no_labels"
        else:
            label_part = f"--labels_file {labels}"

        distributed_part = ""
        if self.gpus > 1:
            # Find a free port which can be used for distributed training
            import socket
            free_port = None
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for port in range(9000, 10000):
                try:
                    sock.bind(('', port))
                    sock.close()
                    free_port = port
                    break
                except OSError:
                    # Try next port
                    pass

            if free_port is None:
                raise Exception("No free port for distributed training available.")
            
            distributed_part =  f"-m torch.distributed.launch --nproc_per_node {self.gpus} --master_port {free_port}"

        preds = self.preds if split_id == -1 else f"preds.{split_id}.json"

        self.sh(
            '{python} {distributed_part} {baseline_path}/baseline.py --checkpoint {checkpoint} {label_part} --output_file {preds} --dataroot {dataroot} --eval_dataset {eval_split} {additional_args}',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            distributed_part=distributed_part,
            checkpoint=self.checkpoint,
            label_part=label_part,
            preds=preds,
            dataroot=data_path,
            eval_split=self.eval_split,
            additional_args=self.additional_args,
        )

    def tasks(self):
        if self.split > 1:
            yield Task('split_labels', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
            yield Task('run', rqmt={'cpu': self.gpus + 1, 'gpu': self.gpus, 'mem': 15, 'time': int(20 / self.gpus + 2 + self.extra_time)}, args=range(self.split))
            yield Task('merge_preds', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
        else:
            yield Task('run', rqmt={'cpu': self.gpus + 1, 'gpu': self.gpus, 'mem': 16, 'time': int(48 / self.gpus + 2 + self.extra_time)})


class ScoreBaselineJob(Job):
    __sis_hash_exclude__ = {
        'data_path': None
    }

    def __init__(self, labels, dataset='val', data_path=None):
        self.labels = labels
        self.dataset = dataset

        if data_path is None:
            self.data_path = pathlib.Path(gs.BASELINE_ROOT) / 'data'
        else:
            self.data_path = data_path

        self.scores = self.output_path('scores.json')

    def run(self):
        self.sh(
            '{python} {baseline_path}/scripts/scores.py --dataset {dataset} --dataroot {dataroot} --outfile {preds} --scorefile {scores}',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            dataset=self.dataset,
            dataroot=self.data_path,
            preds=self.labels,
            scores=self.scores
        )

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 2, 'time': 1}, mini_task=True)


class RunMultitaskOnTrainJob(Job):
    __sis_hash_exclude__ = {
        'config': None,
        'extra_time': 0,
        'data_path': None,
        'eval_split': 'val',
        'split': 1,
        'splits_to_run': None
    }

    def __init__(self, labels=None, data_path=None, eval_split='val', extra_time=0, split=1, splits_to_run=None):
        self.labels = labels

        if data_path is None:
            self.data_path = pathlib.Path(gs.BASELINE_ROOT) / 'data'
        else:
            self.data_path = data_path
        self.eval_split = eval_split
        
        self.preds = self.output_path('preds.json')
        self.gpus = 1
        self.extra_time = extra_time
        
        self.split = split
        self.new_data_path = 'split_data'
        self.splits_to_run = splits_to_run

    def split_labels(self):
        import os
        import json
        import shutil

        knowledge_path = os.path.join(self.data_path, 'knowledge.json')
        log_path = os.path.join(self.data_path, self.eval_split, 'logs.json')
        label_path = self.labels

        with open(label_path) as fp:
            labels = json.load(fp)
        with open(log_path) as fp:
            logs = json.load(fp)
        
        assert len(logs) == len(labels)
        num_labels = len(labels)

        os.mkdir(self.new_data_path)

        for split_id in range(self.split):
            new_data_path = os.path.join(self.new_data_path, str(split_id))
            os.mkdir(new_data_path)

            shutil.copyfile(knowledge_path, os.path.join(new_data_path, 'knowledge.json'))

            new_data_path = os.path.join(new_data_path, self.eval_split)
            os.mkdir(new_data_path)

            low = (num_labels // self.split) * split_id
            upper = (num_labels // self.split) * (split_id + 1) if split_id != (self.split - 1) else num_labels
            
            with open(os.path.join(new_data_path, 'logs.json'), 'w') as fp:
                json.dump(logs[low:upper], fp)

            with open(os.path.join(new_data_path, 'labels.json'), 'w') as fp:
                json.dump(labels[low:upper], fp)

    def merge_preds(self):
        import json
        preds = []
        for split_id in range(self.split):
            with open(f'preds.{split_id}.json') as fp:
                preds += json.load(fp)
        with open(self.preds, 'w') as fp:
            json.dump(preds, fp)

    def run(self, split_id=-1):
        self.sh(
            '/u/daheim/miniconda3/bin/python3 /u/daheim/alexa-with-dstc9-track1-dataset/baseline/baseline.py --checkpoint /u/daheim/alexa-with-dstc9-track1-dataset/setup/work/baseline/TrainBaselineJob.9ywLDTaiGRXO/output/checkpoint --labels_file split_data/{split_id}/train/labels.json --output_file preds.{split_id}.json --dataroot split_data/{split_id} --eval_dataset train --eval_only --eval_all_snippets --multitask --selection --eval_task "selection" --selection_type "all"',
            split_id=split_id
        )

    def tasks(self):
        if self.split > 1:
            yield Task('split_labels', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
            args_range = range(32, self.split) if self.splits_to_run is None else self.splits_to_run
            yield Task('run', rqmt={'cpu': self.gpus + 1, 'gpu': self.gpus, 'mem': 15, 'time': 23 + self.extra_time}, args=args_range)
            yield Task('merge_preds', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
        else:
            yield Task('run', rqmt={'cpu': self.gpus + 1, 'gpu': self.gpus, 'mem': 16, 'time': int(48 / self.gpus + 2 + self.extra_time)})

class Timer(Job):
    __sis_hash_exclude__ = {
        'data_path': None
    }

    def __init__(self, checkpoint, dataset='val', num_samples=1000, data_path=None):
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.num_samples = num_samples

        if data_path is None:
            self.data_path = pathlib.Path(gs.BASELINE_ROOT) / 'data'
        else:
            self.data_path = data_path

        self.times = self.output_path('times.txt')

    def run(self):
        self.sh(
            '{python} {baseline_path}/runtimes.py --checkpoint {checkpoint} --dataset {dataset} --dataroot {dataroot} --num_samples {num_samples} --outfile {outfile}',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            checkpoint=self.checkpoint,
            dataset=self.dataset,
            dataroot=self.data_path,
            num_samples=self.num_samples,
            outfile=self.times
        )

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 1, 'mem': 15, 'time': 4})
