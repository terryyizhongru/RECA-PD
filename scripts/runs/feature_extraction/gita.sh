#!/bin/bash

set -e

dataset_dir=./data/gita/

# -- extract DisVoice features
python scripts/feature_extraction/extract_disvoice_features.py --wav-dir $dataset_dir/norm_audios/ --output-dir $dataset_dir/speech_features/

# -- extract Wav2Vec2.0 features
python scripts/feature_extraction/extract_wav2vec_features.py --wav-dir $dataset_dir/norm_audios/ --output-dir $dataset_dir/speech_features/wav2vec/
