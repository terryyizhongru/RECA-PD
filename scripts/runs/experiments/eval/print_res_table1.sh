#!/bin/bash

set -e

folder=$1
rm $folder/overall_performance_F1.txt || true
touch $folder/overall_performance_F1.txt

for task in SUSTAINED-VOWELS WORDS DDK SENTENCES READ MONOLOGUE; do


python scripts/evaluation/overall_performance_F1.py --exps-dir $folder/${task} >> $folder/overall_performance_F1.txt


done;

cat $folder/overall_performance_F1.txt 
python scripts/evaluation/cal_avg.py $folder/overall_performance_F1.txt 