#!/bin/bash

version="baseline"

# End-to-end evaluation:
# This is a demonstration on how to generate responses with the trained models
# The input files are knowledge.json and logs.json. labels.json is not required
# We use the validation data in this example.

# Prepare directories for intermediate results of each subtask
mkdir -p /work/smt2/daheim/dstc9_baseline/pred/val

# First we do knowledge-seeking turn detection on the test dataset
# Use --eval_dataset to specify the name of the dataset, in this case, val.
# Use --output_file to generate labels.json with predictions
# Specify --no_labels since there's no labels.json to read
python3 /u/daheim/alexa-with-dstc9-track1-dataset/baseline.py --eval_only --checkpoint /work/smt2/daheim/dstc9_baseline/runs/ktd-${version}/ \
   --eval_dataset val \
   --dataroot /u/daheim/alexa-with-dstc9-track1-dataset/data \
   --no_labels \
   --output_file /work/smt2/daheim/dstc9_baseline/pred/val/baseline.ktd.json

# Next we do knowledge selection based on the predictions generated previously
# Use --labels_file to take the results from the previous task
# Use --output_file to generate labels.json with predictions
python3 /u/daheim/alexa-with-dstc9-track1-dataset/baseline.py --eval_only --checkpoint /work/smt2/daheim/dstc9_baseline/runs/ks-all-${version} \
   --eval_all_snippets \
   --dataroot /u/daheim/alexa-with-dstc9-track1-dataset/data \
   --eval_dataset val \
   --labels_file /work/smt2/daheim/dstc9_baseline/pred/val/baseline.ktd.json \
   --output_file /work/smt2/daheim/dstc9_baseline/pred/val/baseline.ks.json

# Finally we do response generation based on the selected knowledge
python3 /u/daheim/alexa-with-dstc9-track1-dataset/baseline.py --generate /work/smt2/daheim/dstc9_baseline/runs/rg-hml128-kml128-${version} \
        --generation_params_file /u/daheim/alexa-with-dstc9-track1-dataset/baseline/configs/generation/generation_params.json \
        --eval_dataset val \
        --dataroot /u/daheim/alexa-with-dstc9-track1-dataset/data \
        --labels_file /work/smt2/daheim/dstc9_baseline/pred/val/baseline.ks.json \
        --output_file /work/smt2/daheim/dstc9_baseline/baseline_val.json
