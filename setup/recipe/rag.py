import json

from sisyphus import *


class CreateRagCheckpoint(Job):
    __sis_hash_exclude__ = {
        'new_hash': False
    }

    def __init__(self, encoder_model_name_or_path, generator_model_name_or_path, new_hash=False):
        self.encoder_model_name_or_path = encoder_model_name_or_path
        self.generator_model_name_or_path = generator_model_name_or_path
        self.checkpoint = self.output_path('checkpoint', directory=True)

    def run(self):
        self.sh(
            '{python} {baseline_path}/create_rag_checkpoint.py "{generator_model}" "{encoder_model}" "{output_checkpoint}"',
            baseline_path=gs.BASELINE_ROOT,
            python=gs.BASELINE_PYTHON_EXE,
            generator_model=self.generator_model_name_or_path,
            encoder_model=self.encoder_model_name_or_path,
            output_checkpoint=self.checkpoint,
        )

    def tasks(self):
        yield Task('run', rqmt={'cpu': 1, 'gpu': 0, 'mem': 8, 'time': 1})
