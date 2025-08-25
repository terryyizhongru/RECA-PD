import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from distutils.util import strtobool

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.legend_handler import HandlerTuple

from scipy.stats import pearsonr
from scipy.stats import ks_2samp

def preprocess_data_paths(analysis_dir):
    data_paths = glob.glob(f'{analysis_dir}/**/*.csv', recursive=True)

    data_paths_df = pd.DataFrame(data_paths, columns=['data_path'])
    data_paths_df['phoneme_id'] = data_paths_df['data_path'].map(lambda x: os.path.basename(x).split('_')[1])

    return data_paths_df, list(data_paths_df['phoneme_id'].unique())

def main(args):

    data_paths_df, phoneme_set = preprocess_data_paths(args.analysis_dir)

    for phoneme_id in tqdm(phoneme_set, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

        phoneme_data_dfs = []
        phoneme_data_df_paths = data_paths_df[data_paths_df['phoneme_id'] == phoneme_id]['data_path'].tolist()
        for phoneme_data_df_path in phoneme_data_df_paths:
            phoneme_data_df = pd.read_csv(phoneme_data_df_path)
            phoneme_data_dfs.append( phoneme_data_df )

        phoneme_data = pd.concat(phoneme_data_dfs)

        # -- -- targeting a specific disease stadium
        updrs_speech_values = sorted(list(phoneme_data['UPDRS Scale'].unique()))

        # -- plotting data
        plt.figure()

        output_path = os.path.join(args.output_dir, 'figures', f'phoneme_{phoneme_id}_analysis.png')
        sns_plot = sns.boxplot(data=phoneme_data, x='Informed High-Level Speech Dimension', y=f'Attention to phoneme /{phoneme_id}/', hue='UPDRS Scale', hue_order=updrs_speech_values, palette='tab10', boxprops={'alpha': 0.4})
        sns.stripplot(data=phoneme_data, x='Informed High-Level Speech Dimension', y=f'Attention to phoneme /{phoneme_id}/', hue='UPDRS Scale', hue_order=updrs_speech_values, dodge=True, palette='tab10', ax=sns_plot)

        handles, labels = sns_plot.get_legend_handles_labels()
        lgd = sns_plot.legend(
          title='UPDRS Scale',
          handles=[(handles[i], handles[i+len(updrs_speech_values)]) for i in range(len(updrs_speech_values))],
          labels=updrs_speech_values,
          loc='upper left', handlelength=4, bbox_to_anchor=(1, 1),
          handler_map={tuple: HandlerTuple(ndivide=None)},
        )

        sns_plot.figure.savefig(output_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Cross Temporal Phoneme Overall Attention Score Analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--analysis-dir', required=True, type=str, help='Directory where per-sample phoneme analysis can be retrieved.')
    parser.add_argument('--output-dir', default='./plots/cross_full/overall_phoneme_analysis/high-level/', type=str, help='Output directory where plots will be saved.')

    args = parser.parse_args()

    # -- creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['figures', 'data']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # -- main script execution
    main(args)
