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
    def __init__(self, name, config, additional_args="", gpus=2):
        self.name = name
        self.additional_args = additional_args
        self.gpus = gpus

        baseline_path = pathlib.Path(gs.BASELINE_ROOT)
        self.data_path = baseline_path / 'data'

        self.config = config

        self.checkpoint = self.output_path('checkpoint', directory=True)

    def write_config(self):
        with open('params.json', 'w') as fp:
            json.dump(self.config, fp)

    def run(self):
        import os
        import pathlib

        runs_path = pathlib.Path('runs')
        runs_path.mkdir(parents=True, exist_ok=True)
        os.symlink(src=self.checkpoint,
                   dst=runs_path / self.name,
                   target_is_directory=True)

        self.sh(
            '{python} -m torch.distributed.launch --nproc_per_node {num_gpus} {baseline_path}/baseline.py --dataroot {dataroot} --exp_name {name} --params_file params.json {additional_args}',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            num_gpus=self.gpus,
            dataroot=self.data_path,
            additional_args=self.additional_args,
            name=self.name,
        )

    def tasks(self):
        yield Task('write_config', rqmt={'cpu': 1, 'gpu': 0, 'mem': 1, 'time': 1}, mini_task=True)
        yield Task('run', rqmt={'cpu': self.gpus, 'gpu': self.gpus, 'mem': 32, 'time': 24})


class RunBaselineJob(Job):
    def __init__(self, checkpoint, labels=None, additional_args="", gpus=1):
        self.checkpoint = checkpoint

        self.labels = labels
        self.additional_args = additional_args
        self.data_path = pathlib.Path(gs.BASELINE_ROOT) / 'data'
        
        self.preds = self.output_path('preds.json')
        self.gpus = gpus

    def run(self):
        if self.labels is None:
            label_part = "--no_labels"
        else:
            label_part = f"--labels_file {self.labels}"

        distributed_part = ""
        if self.gpus > 1:
            distributed_part =  f"-m torch.distributed.launch --nproc_per_node {self.gpus}"

        self.sh(
            '{python} {distributed_part} {baseline_path}/baseline.py --checkpoint {checkpoint} {label_part} --output_file {preds} --dataroot {dataroot} --eval_dataset val {additional_args}',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            distributed_part=distributed_part,
            checkpoint=self.checkpoint,
            label_part=label_part,
            preds=self.preds,
            dataroot=self.data_path,
            additional_args=self.additional_args,
        )

    def tasks(self):
        yield Task('run', rqmt={'cpu': self.gpus + 1, 'gpu': self.gpus, 'mem': 16, 'time': 24})

class ScoreBaselineJob(Job):
    def __init__(self, labels, dataset='val'):
        self.labels = labels
        self.dataset = dataset

        self.data_path = pathlib.Path(gs.BASELINE_ROOT) / 'data'

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
