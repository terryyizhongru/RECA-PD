#!/bin/bash

set -e

dataset=gita_splits2

for task in combined_set; do
for seed in 12 21 33 42 52; do
for f in 0 1 2 3 4 5 6 7 8 9; do

  CUDA_VISIBLE_DEVICES=0 python scripts/model_pipeline/pipeline.py \
      --config ./configs/framework.yaml \
      --training-dataset ./splits/$dataset/fold_$f/train.csv \
      --validation-dataset ./splits/$dataset/fold_$f/test.csv \
      --test-dataset ./splits/$dataset/fold_$f/test.csv \
      --output-dir ./exps/$dataset/M4/${task}/seed${seed}/fold_${f}/ \
      --yaml-overrides device:cuda seed:$seed model:RECAPD \
      --save-attention-scores False

done;
done;
done;
