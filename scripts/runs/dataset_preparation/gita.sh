#!/bin/bash

set -e

original_dataset_dir=$1
path_to_original_metadata=$2

# -- preparing directory where to store the data
k_folds=5
dataset_dir=./data/gita # ; mkdir -p $dataset_dir;
splits_dir=./splits/gita # ; mkdir -p $splits_dir;

# -- original audio samples restructuring
python scripts/dataset_splitting/gita_restructure.py --data-dir $original_dataset_dir --metadata-path $path_to_original_metadata --new-data-dir $dataset_dir/audios/

# -- audio normalization preprocessing
python scripts/feature_extraction/wav_preprocessing.py --wav-dir $dataset_dir/audios/ --output-dir $dataset_dir/norm_audios/

# -- dataset split definition
python scripts/dataset_splitting/gita_dataset.py --samples-dir $dataset_dir/norm_audios/ --metadata-path $path_to_original_metadata --output-dir $splits_dir

# -- stratified cross-validation splits
python scripts/dataset_splitting/cross_validation_splitting.py --dataset-path $splits_dir/dataset.csv --k-folds $k_folds


