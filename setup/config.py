import sys
import copy
import os
from math import sqrt
import pathlib
import json

from recipe.baseline import RunMultitaskOnTrainJob, RerunSelectionSplit

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

from sisyphus import *
import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

Path = tk.Path

# ------------------------------ Recipes --------------------------------------

from recipe.baseline import TrainBaselineJob, RunBaselineJob, ScoreBaselineJob
from recipe.data import CreateLabelDifference, MakeTrueTargets, MergeSelectionLabels


async def async_main():
    # Baseline models
    from config_baseline import main as baseline_main
    baseline_main()

    # Hierarchical Selection models
    from config_hierarchical import main as hierarchical_main
    hierarchical_main()

    # Multi-task models
    from config_multitask import main as multitask_main
    multitask_main()

    # Retrieval Augmented Detection experiments
    from config_retrieval_detection_embedding_nll import main as retrieval_detection_main
    retrieval_detection_main()

    # Dense Knowledge Retrieval
    from config_ra_embedding import main as rag_embedding_main
    rag_embedding_main()

    from config_times import main as timer_main
    timer_main()

async def py():
    await async_main()
