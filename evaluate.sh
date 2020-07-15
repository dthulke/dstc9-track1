#!/bin/bash

score_file=baseline_val.score.json

cd /u/daheim/alexa-with-dstc9-track1-dataset

python3 scripts/scores.py --dataset val --dataroot data/ --outfile /work/smt2/daheim/dstc9_baseline/baseline_val.json --scorefile $score_file

cat $score_file