#!/bin/bash

set -e

dataset=gita_splits1

for task in SUSTAINED-VOWELS WORDS DDK SENTENCES MONOLOGUE READ; do
for seed in 12 21 33 42 52; do
for f in 0 1 2 3 4; do

  CUDA_VISIBLE_DEVICES=0 python scripts/model_pipeline/pipeline.py \
      --config ./configs/framework.yaml \
      --training-dataset ./splits/$dataset/fold_$f/fulltrain.csv \
      --validation-dataset ./splits/$dataset/fold_$f/test.csv \
      --test-dataset ./splits/$dataset/fold_$f/test.csv \
      --filter-tasks $task \
      --output-dir ./exps/$dataset/M1/${task}/seed${seed}/fold_${f}/ \
      --yaml-overrides device:cuda seed:$seed model:cross_full \
      --save-attention-scores False

done;
done;
done;
