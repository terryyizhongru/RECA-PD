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
    parser = argparse.ArgumentParser(description='Prepare GITA dataset into a CSV', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--samples-dir', required=True, type=str)
    parser.add_argument('--metadata-path', required=True, type=str)
    parser.add_argument('--output-dir', default='./splits/gita/', type=str)
    args = parser.parse_args()

    # -- loading and processing metadata
    # -- original column IDs: RECODING ORIGINAL NAME,UPDRS,UPDRS-speech,H/Y,SEX,AGE,time after diagnosis
    metadata = pd.read_csv(args.metadata_path)
    metadata['label'] = metadata['RECODING ORIGINAL NAME'].map(lambda x: 1 if 'AVPEPUDEAC' not in x else 0)
    metadata['group_id'] = metadata['label'].map(lambda x: 'PD' if x else 'HC')
    metadata['sex'] = metadata['SEX'].map(lambda x: 1 if x == 'M' else 0)

    # -- processing audio samples
    dataset = []
    wavs = glob.glob(os.path.join(args.samples_dir, '*.wav'), recursive=True)
    for wav_path in tqdm(wavs):

        sample_id = wav_path.split(os.path.sep)[-1].split('.')[0]
        subject_id = re.findall(r"(AVPEPUDEA(?:C)?\d{4})", sample_id)[0]
        task_id = os.path.basename(wav_path).split('_')[1].upper()

        # -- retriveing metadata per sample
        sample = metadata[metadata['RECODING ORIGINAL NAME'] == subject_id]

        sex = sample['sex'].values[0]
        age = sample['AGE'].values[0]
        label = sample['label'].values[0]
        group_id = sample['group_id'].values[0].upper()
        time_after_diagnosis = sample['time after diagnosis'].values[0]
        updrs = sample['UPDRS'].values[0]
        updrs_speech = sample['UPDRS-speech'].values[0]
        hy_scale = sample['H/Y'].values[0]

        # -- preparing the dataset
        dataset.append( (subject_id, sample_id, task_id, label, group_id, sex, age, updrs, updrs_speech, hy_scale, time_after_diagnosis) )


    # -- building the dataset dataframe
    dataset_df = pd.DataFrame(dataset, columns=['subject_id', 'sample_id', 'task_id', 'label', 'group_id', 'sex', 'age', 'updrs_scale', 'updrs_speech', 'hy_scale', 'time_after_diagnosis'])

    # -- adding information about speech features paths
    dataset_dir = os.path.sep.join(args.samples_dir.split(os.path.sep)[:-2])
    for feature_type in ['disvoice/articulation', 'disvoice/glottal', 'disvoice/phonation', 'disvoice/prosody', 'wav2vec/layer07']:
        feature_samples = []

        for i, sample in dataset_df.iterrows():
            sample_path = os.path.join(dataset_dir, 'speech_features', feature_type, f'{sample["sample_id"]}.npz')
            feature_samples.append(sample_path)

        dataset_df[feature_type.replace('/', '').replace('disvoice', '').replace('layer07', '')] = feature_samples

    # -- saving dataset split
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_df.to_csv(os.path.join(args.output_dir, 'dataset.csv'))
