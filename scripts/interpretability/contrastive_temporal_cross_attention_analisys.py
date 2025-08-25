import os
import glob
import argparse
import textgrids
import numpy as np
from utils import *
from dtw_alignment import *
from distutils.util import strtobool

import pandas as pd
import seaborn as sns
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler, StandardScaler

word_emphasis = {
    'VISTE': ['ganar'],
    'JUAN': ['rompió', 'pierna', 'moto'],
    'TRISTE': ['morir'],
    'CUPADO': ['hablar'],
    'OMAR': [],
    'LUISA': [],
    'LAURA': [],
    'MICASA': [],
    'LOSLIBROS': [],
    'ROSITA': [],
}

def contrastive_analysis(attn_scores, informed_metadata_bounds):

    # -- time-wise accumulative relevance per high-level feature type
    lineplot_mean_data = {}

    total_mean_attribution = np.zeros(attn_scores.shape[0])
    for metadata_id, (left_bound, right_bound) in informed_metadata_bounds.items():
        informed_attn_scores = attn_scores[:, left_bound:right_bound]

        attribution_mean_scores = informed_attn_scores.mean(axis=-1)
        lineplot_mean_data[r'$\bf{'+metadata_id+'}$'] = attribution_mean_scores

    return lineplot_mean_data

def create_contrastive_plot(lineplot_mean_data, output_path, phoneme_alignment, word_level=False, word_emphasis=[], sample_rate=49):

    # -- converting time to seconds, taking into account Wav2Vec processes speech at 49Hz
    first_key = list(lineplot_mean_data.keys())[0]
    lineplot_mean_data['Time (seconds)'] = np.arange(lineplot_mean_data[first_key].shape[-1]) / sample_rate

    # -- excluding possible silences at the beginning and end of the utterance
    # if phoneme_alignment[-1].text == '':
    #     xlim = round(phoneme_alignment[-1].xmin * sample_rate)
    #     for k in lineplot_mean_data.keys():
    #         lineplot_mean_data[k] = lineplot_mean_data[k][:xlim].copy()
    #     phoneme_alignment.pop(-1)

    # -- plots
    plt.figure()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(21,7.0), gridspec_kw={'width_ratios': [3, 1], 'wspace':0.115})

    # ax[0].grid()
    # ax[0].set_ylim(-1, 100)

    xaxis_title = 'Time (seconds)'
    data_mean_df = pd.DataFrame(lineplot_mean_data)
    data_mean_df = pd.melt(data_mean_df, id_vars=[xaxis_title])
    data_mean_df = data_mean_df.rename(columns={'value': 'Attention-based Relevance Score', 'variable': 'High-Level Speech Dimension'})

    time_lineplot = sns.lineplot(ax=ax[0], data=data_mean_df, x='Time (seconds)', y='Attention-based Relevance Score', hue='High-Level Speech Dimension', palette="tab10", linewidth=1.5)

    for line in time_lineplot.lines:
        x, y = line.get_xydata().T
        time_lineplot.fill_between(x, 0, y, color=line.get_color(), alpha=.33)

    time_lineplot.get_legend().set_title(None)
    # time_lineplot.set(xticklabels=[])
    # time_lineplot.tick_params(bottom=False)

    # -- delimiting phoneme boundings
    xticks = []
    xticklabels = []
    for i, phoneme in enumerate(phoneme_alignment):
        xmin = phoneme.xmin
        xmax = phoneme.xmax # min(phoneme.xmax, lineplot_mean_data['Time (seconds)'][-1])

        time_lineplot.axvline(xmin, color='gray', linestyle='--')
        time_lineplot.axvline(xmax, color='gray', linestyle='--')

        middle_point = (xmin + xmax) / 2
        xticks.append(middle_point)
        xticklabels.append(r'$\bf/{'+phoneme.text+'}/$') # '\n'+str(round(middle_point,2)))
        # time_lineplot.text(middle_point, -.05, f'/{phoneme.text}/')

    # -- just to highligh the emphasis, target words for prosody
    if word_level:
        previous_lim = 0
        for i, phoneme in enumerate(phoneme_alignment):
            if phoneme.text in word_emphasis:
                time_lineplot.axvspan(previous_lim, phoneme.xmin, color='white', alpha=0.6, zorder=2)

                time_lineplot.axvline(phoneme.xmin, color='black', linestyle='--', zorder=3)
                time_lineplot.axvline(phoneme.xmax, color='black', linestyle='--', zorder=3)

                previous_lim = phoneme.xmax

        time_lineplot.axvspan(previous_lim, lineplot_mean_data['Time (seconds)'][-1], color='white', alpha=0.6, zorder=100)

    prelast_lim = phoneme_alignment[-2].xmax
    last_lim = phoneme_alignment[-1].xmax
    print('AQUI:', prelast_lim, last_lim, lineplot_mean_data['Time (seconds)'][-1])
    time_lineplot.axvspan(prelast_lim, last_lim, color='white', alpha=1.0, zorder=100)
    time_lineplot.axvline(prelast_lim, color='black' if word_level else 'gray', linestyle='--', zorder=200)
    time_lineplot.set_xlim(left=-0.02, right=prelast_lim+0.02)
    time_lineplot.set_xticks(xticks[:-1])
    time_lineplot.set_xticklabels(xticklabels[:-1]) # , rotation=30)

    if word_level:
        for i, t in enumerate(time_lineplot.get_xticklabels()):
            if i not in [2,4,9]:
                t.set_alpha(0.6)

    time_lineplot.set_xlabel('')

    data_mean_df = data_mean_df.drop(xaxis_title, axis=1)
    data_mean_df.to_csv(output_path.replace('figures', 'stats').replace('.png', '.csv'), index=False)
    data_total_df = data_mean_df.groupby('High-Level Speech Dimension').sum()

    # ax[1].tick_params("x", labelrotation=30)
    total_barplot = sns.barplot(ax=ax[1], x=data_total_df.index, y=data_total_df['Attention-based Relevance Score'])
    total_barplot.set_xticklabels(rotation=30, labels=data_total_df.index)

    sns.move_legend(time_lineplot, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
    time_lineplot.figure.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight', format='svg')
    plt.close()

def load_data_by_condition(args, filter_by_condition):
    sample_ids, attn_scores, informed_metadata = load_model_output(
        args.exps_dir,
        args.filter_by_id,
        filter_by_condition,
        attn_scores_id='cross_time_mha_scores',
    )

    print(f'\nConsidering {len(attn_scores)} {"HC" if filter_by_condition == 0 else "PD"} samples based on the ID filter: {args.filter_by_id}')

    return sample_ids, attn_scores, informed_metadata

def main(args):

    # -- loading phoneme-level alignment
    phoneme_alignment = textgrids.TextGrid(args.phoneme_alignment)['words' if args.word_level else 'phones']

    # -- loading target model outputs and informed speech features metadata
    hc_sample_ids, hc_attn_scores, _ = load_data_by_condition(args, 0)
    pd_sample_ids, pd_attn_scores, informed_metadata = load_data_by_condition(args, 1)

    # -- applying dynamic time warping for all samples including both conditions
    sample_ids = hc_sample_ids + pd_sample_ids
    attn_scores = hc_attn_scores + pd_attn_scores
    aligned_attn_scores = apply_dtw_alignment(args.wav2vec_dir, args.task_id, sample_ids, attn_scores)

    hc_aligned_attn_scores = aligned_attn_scores[:len(hc_sample_ids)]
    pd_aligned_attn_scores = aligned_attn_scores[len(hc_sample_ids):]

    # -- computing average across attention weights
    hc_attn_scores_avg = hc_aligned_attn_scores.mean(axis=0)

    # -- high-level feature attribution analysis
    # -- retrieving informed speech feature metadata information
    _, informed_metadata_bounds = informed_metadata

    all_contrastive_attributions = {r'$\bf{'+metadata_id+'}$': np.empty((0, hc_attn_scores_avg.shape[0])) for metadata_id, _ in informed_metadata_bounds.items()}
    for pd_sample_id, pd_attn_score in zip(pd_sample_ids, pd_aligned_attn_scores):

        contrastive_pd_attn_score = pd_attn_score - hc_attn_scores_avg

        # -- relevance thresholding
        thr = np.quantile(contrastive_pd_attn_score, 0.95)
        contrastive_pd_attn_score[contrastive_pd_attn_score < thr] = 1e-10

        sample_contrastive_attributions = contrastive_analysis(contrastive_pd_attn_score, informed_metadata_bounds)

        for metadata_id in sample_contrastive_attributions.keys():
            all_contrastive_attributions[metadata_id] = np.vstack((
                all_contrastive_attributions[metadata_id],
                sample_contrastive_attributions[metadata_id],
            ))

    # -- normalizing attributions
    attribution_keys = list(all_contrastive_attributions.keys())

    all_attributions_together_per_sample = np.vstack((
        all_contrastive_attributions[attribution_keys[0]].T,
        all_contrastive_attributions[attribution_keys[1]].T,
        all_contrastive_attributions[attribution_keys[2]].T,
        all_contrastive_attributions[attribution_keys[3]].T,
    ))

    scaler = MinMaxScaler(feature_range=(0, 1)) # StandardScaler()
    scaler.fit(all_attributions_together_per_sample)
    scaled_attributions = scaler.transform(all_attributions_together_per_sample)

    for i, metadata_id in enumerate(all_contrastive_attributions.keys()):
        attribution_splits = np.split(scaled_attributions, len(attribution_keys), axis=0)
        all_contrastive_attributions[metadata_id] = attribution_splits[i].T

    # -- creating contrastive plots
    for sample_idx, pd_sample_id in enumerate(pd_sample_ids):
        sample_contrastive_attributions = {}

        # -- recovering sample-level attributions
        for metadata_id, metadata_attributions in all_contrastive_attributions.items():
            sample_contrastive_attributions[metadata_id] = metadata_attributions[sample_idx]

        output_path = os.path.join(args.output_dir, 'figures', f'{pd_sample_id}.png')
        create_contrastive_plot(sample_contrastive_attributions, output_path, phoneme_alignment, word_level=args.word_level, word_emphasis=word_emphasis[args.filter_by_id] if args.task_id in ['SENTENCES'] else [])
        plt.close()

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Cross Temporal Attention Score Analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where model output can be retrieved.')
    parser.add_argument('--wav2vec-dir', required=True, type=str, help='Directory where wav2vec embeddings can be retrieved.')
    parser.add_argument('--task-id', required=True, type=str, help='We thought was not necessary, but it is.')
    parser.add_argument('--filter-by-id', default='', type=str, help='In case you do not want the average across subjects, but only for a specific subset of specific sample')
    parser.add_argument('--phoneme-alignment', default='', type=str, help='TextGrid specifying the phoneme-level alignment for the minimum-length audio sample')
    parser.add_argument('--word-level', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--output-dir', default='./plots/cross_ssl_full/contrasting_time/', type=str, help='Output directory where plots will be saved.')

    args = parser.parse_args()

    # -- creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['figures', 'stats']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # -- main script execution
    main(args)
