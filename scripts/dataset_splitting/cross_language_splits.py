import os
import argparse
import pandas as pd

def gathering_datasets(dataset_paths, output_path):

    datasets = []
    for dataset_path in dataset_paths:
        dataset_df = pd.read_csv(dataset_path)
        focused_df = dataset_df[[
            'subject_id',
            'sample_id',
            'task_id',
            'label',
            'wav2vec',
            'prosody_static_disvoice',
            'phonation_static_disvoice',
            'articulation_static_disvoice',
            'glottal_static_disvoice',
        ]].copy()

        print(len(focused_df))
        datasets.append( focused_df )

    gathered_dataset = pd.concat(datasets)
    gathered_dataset.to_csv(output_path, index=False)

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='Prepare splits for cross-lingual experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-paths', nargs='+', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    gathering_datasets(args.dataset_paths, args.output_path)
