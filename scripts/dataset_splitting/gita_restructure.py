import os
import re
import glob
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from distutils.util import strtobool

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='Restructure the original GITA audio samples', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--metadata-path', required=True, type=str)
    parser.add_argument('--new-data-dir', default='./data/gita/audios/', type=str)
    args = parser.parse_args()

    os.makedirs(args.new_data_dir, exist_ok=True)

    # -- loading and processing metadata
    # -- original column IDs: RECODING ORIGINAL NAME,UPDRS,UPDRS-speech,H/Y,SEX,AGE,time after diagnosis
    metadata = pd.read_csv(args.metadata_path)
    metadata['label'] = metadata['RECODING ORIGINAL NAME'].map(lambda x: 1 if 'AVPEPUDEAC' not in x else 0)
    metadata['group_id'] = metadata['label'].map(lambda x: 'PD' if x else 'HC')

    # -- processing audio samples
    ignored_samples = 0
    wavs = glob.glob(os.path.join(args.data_dir, '**/*.wav'), recursive=True)
    for wav_path in tqdm(wavs):
        if 'los_que_' in wav_path or 'las_que_' in wav_path:
            ignored_samples += 1
            continue

        sample_id = wav_path.split(os.path.sep)[-1].split('.')[0]
        subject_id = re.findall(r"(AVPEPUDEA(?:C)?\d{4})", sample_id)[0]
        if subject_id in metadata['RECODING ORIGINAL NAME'].tolist():
            sample_id = f'{subject_id}_{sample_id.split(subject_id)[-1]}'.replace('_PRECUPADO', '_PREOCUPADO').replace('__', '_').upper()
            task_id = wav_path.split(args.data_dir)[-1].split(os.path.sep)[0].upper()
            task_id = task_id.replace('MODULATED VOWELS', 'MODULATED-VOWELS').replace('SENTENCES2', 'SENTENCES').replace('DDK ANALYSIS', 'DDK').replace('READ TEXT', 'READ-TEXT')

            # -- retriveing metadata per sample
            sample = metadata[metadata['RECODING ORIGINAL NAME'] == subject_id]
            group_id = sample['group_id'].values[0].upper()

            new_wav_path = os.path.join(args.new_data_dir, f'{group_id}_{task_id}_{sample_id}.wav').replace('_VOWELS_', '_SUSTAINED-VOWELS_').replace('_PRECUPADO', '_PREOCUPADO').replace('__', '_')
            shutil.copy(wav_path, new_wav_path)

    print(f'{ignored_samples} samples were ignored because of some of the subjects were not further considered in the study.')
