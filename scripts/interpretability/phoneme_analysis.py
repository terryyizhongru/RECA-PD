import os
import glob
import argparse
import textgrids
import numpy as np
from tqdm import tqdm
from distutils.util import strtobool

from utils import *
from dtw_alignment import *

import pandas as pd
import seaborn as sns
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy.stats import ks_2samp

def load_data_by_condition(args, filter_by_condition):
    sample_ids, attn_scores, informed_metadata = load_model_output(
        args.exps_dir,
        args.filter_by_id,
        filter_by_condition,
        attn_scores_id='cross_time_mha_scores',
    )

    print(f'\nConsidering {len(attn_scores)} {"HC" if filter_by_condition == 0 else "PD"} samples based on the ID filter: {args.filter_by_id}')

    return sample_ids, attn_scores, informed_metadata

def focused_contrastive_preprocessing(pd_sample_ids, pd_aligned_attn_scores, hc_attn_scores_avg, informed_metadata_ids):
    all_contrastive_attributions = {r'$\bf{'+metadata_id+'}$': np.empty((0, hc_attn_scores_avg.shape[0])) for metadata_id in informed_metadata_ids}

    # -- for each PD audio sample in the dataset
    for pd_sample_id, pd_attn_score in zip(pd_sample_ids, pd_aligned_attn_scores):

        # -- contrast w.r.t. the average attention scores from control population
        contrastive_pd_attn_score = pd_attn_score - hc_attn_scores_avg

        # -- relevance attention score thresholding
        # thr = np.quantile(contrastive_pd_attn_score, 0.95)
        # contrastive_pd_attn_score[contrastive_pd_attn_score < thr] = 1e-10

        # -- time-wise accumulative relevance per high-level feature type
        sample_contrastive_attributions = {}
        for idx, metadata_id in enumerate(informed_metadata_ids):
            attribution_attn_scores = contrastive_pd_attn_score[:, idx]
            sample_contrastive_attributions[r'$\bf{'+metadata_id+'}$'] = attribution_attn_scores

        for metadata_id in sample_contrastive_attributions.keys():
            all_contrastive_attributions[metadata_id] = np.vstack((
                all_contrastive_attributions[metadata_id],
                sample_contrastive_attributions[metadata_id],
            ))

    # -- normalizing attributions
    all_attributions_together_per_sample = np.vstack([
        all_contrastive_attributions[metadata_id].T
        for metadata_id in all_contrastive_attributions.keys()
    ])

    scaler = MinMaxScaler(feature_range=(0, 1)) # StandardScaler()
    scaled_attributions = scaler.fit_transform(all_attributions_together_per_sample)

    for i, metadata_id in enumerate(all_contrastive_attributions.keys()):
        attribution_splits = np.split(scaled_attributions, len(all_contrastive_attributions.keys()), axis=0)
        all_contrastive_attributions[metadata_id] = attribution_splits[i].T

    return all_contrastive_attributions

def contrastive_preprocessing(pd_sample_ids, pd_aligned_attn_scores, hc_attn_scores_avg, informed_metadata_bounds):
    all_contrastive_attributions = {r'$\bf{'+metadata_id+'}$': np.empty((0, hc_attn_scores_avg.shape[0])) for metadata_id, _ in informed_metadata_bounds.items()}

    # -- for each PD audio sample in the dataset
    for pd_sample_id, pd_attn_score in zip(pd_sample_ids, pd_aligned_attn_scores):

        # -- contrast w.r.t. the average attention scores from control population
        contrastive_pd_attn_score = pd_attn_score - hc_attn_scores_avg

        # -- relevance attention score thresholding
        thr = np.quantile(contrastive_pd_attn_score, 0.95)
        contrastive_pd_attn_score[contrastive_pd_attn_score < thr] = 1e-10

        # -- time-wise accumulative relevance per high-level feature type
        sample_contrastive_attributions = {}
        for metadata_id, (left_bound, right_bound) in informed_metadata_bounds.items():
            informed_attn_scores = contrastive_pd_attn_score[:, left_bound:right_bound]

            attribution_mean_scores = informed_attn_scores.mean(axis=-1)
            sample_contrastive_attributions[r'$\bf{'+metadata_id+'}$'] = attribution_mean_scores

        for metadata_id in sample_contrastive_attributions.keys():
            all_contrastive_attributions[metadata_id] = np.vstack((
                all_contrastive_attributions[metadata_id],
                sample_contrastive_attributions[metadata_id],
            ))

    # -- normalizing attributions
    all_attributions_together_per_sample = np.vstack([
        all_contrastive_attributions[metadata_id].T
        for metadata_id in all_contrastive_attributions.keys()
    ])

    scaler = MinMaxScaler(feature_range=(0, 1)) # StandardScaler()
    scaled_attributions = scaler.fit_transform(all_attributions_together_per_sample)

    for i, metadata_id in enumerate(all_contrastive_attributions.keys()):
        attribution_splits = np.split(scaled_attributions, len(all_contrastive_attributions.keys()), axis=0)
        all_contrastive_attributions[metadata_id] = attribution_splits[i].T

    return all_contrastive_attributions

def phoneme_alignment_preprocessing(phoneme_alignments, sample_rate=49, study_coarticulations=False):
    processed_phoneme_alignments = {}

    if not study_coarticulations:

        for phone in phoneme_alignments:

            if not phone.text:
                continue

            min_frame = round(phone.xmin * sample_rate)
            max_frame = round(phone.xmax * sample_rate)

            if phone.text not in processed_phoneme_alignments:
                processed_phoneme_alignments[phone.text] = [(min_frame, max_frame)]
            else:
                processed_phoneme_alignments[phone.text].append( (min_frame, max_frame) )

    else:
        phone_set = [(i, phone.text) for i, phone in enumerate(phoneme_alignments) if phone.text]

        window_size = 3
        for i in range(len(phone_set) - window_size + 1):
            triphoneme = ''.join([phone for _, phone in phone_set[i: i + window_size]])

            min_actual_idx = phone_set[i][0]
            min_frame = round(phoneme_alignments[min_actual_idx].xmin * sample_rate)

            max_actual_idx = phone_set[i+window_size-1][0]
            max_frame = round(phoneme_alignments[max_actual_idx].xmax * sample_rate)

            if triphoneme not in processed_phoneme_alignments:
                processed_phoneme_alignments[triphoneme] = [(min_frame, max_frame)]
            else:
                processed_phoneme_alignments[triphoneme].append( (min_frame, max_frame) )

    return processed_phoneme_alignments

def phoneme_analysis(sample_ids, contrastive_attributions, target_idxs, phoneme_segments, variable):

    phoneme_data = []
    for metadata_id in contrastive_attributions.keys():
        metadata_attributions = contrastive_attributions[metadata_id] # -- (nsamples, time)

        # -- indexing the target audio samples
        target_sample_ids = np.array(sample_ids)[target_idxs]
        target_metadata_attributions = metadata_attributions[target_idxs]

        for segment_id, (min_frame, max_frame) in enumerate(phoneme_segments):

            phoneme_attributions = target_metadata_attributions[:, min_frame:max_frame]

            # TODO: Discuss. Max for Isolated Phones? Mean for Triphonemes?
            # max_phoneme_attribution = phoneme_attributions.max(axis=-1)
            max_phoneme_attribution = phoneme_attributions.mean(axis=-1)

            for idx, max_sample_phoneme_attribution in enumerate(max_phoneme_attribution):
                phoneme_data.append( (target_sample_ids[idx], segment_id, str(variable), metadata_id, max_sample_phoneme_attribution) )

    return phoneme_data

def main(args):

    # -- loading target model outputs and informed speech features metadata
    hc_sample_ids, hc_attn_scores, _ = load_data_by_condition(args, 0)
    pd_sample_ids, pd_attn_scores, informed_metadata = load_data_by_condition(args, 1)

    # -- creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.focused_study:
        metadata_ids = informed_metadata[0]
        for metadata_id in metadata_ids:
            os.makedirs(os.path.join(args.output_dir, metadata_id, 'figures'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, metadata_id, 'data'), exist_ok=True)
    else:
        for subdir in ['figures', 'data']:
            os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # -- applying dynamic time warping for all samples including both conditions
    sample_ids = hc_sample_ids + pd_sample_ids
    attn_scores = hc_attn_scores + pd_attn_scores
    aligned_attn_scores = apply_dtw_alignment(args.wav2vec_dir, args.task_id, sample_ids, attn_scores)

    hc_aligned_attn_scores = aligned_attn_scores[:len(hc_sample_ids)]
    pd_aligned_attn_scores = aligned_attn_scores[len(hc_sample_ids):]

    # -- computing average across attention weights
    hc_attn_scores_avg = hc_aligned_attn_scores.mean(axis=0)

    # -- retrieving informed speech feature metadata information
    informed_metadata_ids, informed_metadata_bounds = informed_metadata

    # -- applying contrastive preprocessing w.r.t. the average of the control group
    if args.focused_study:
        all_contrastive_attributions = focused_contrastive_preprocessing(pd_sample_ids, pd_aligned_attn_scores, hc_attn_scores_avg, informed_metadata_ids)
    else:
        all_contrastive_attributions = contrastive_preprocessing(pd_sample_ids, pd_aligned_attn_scores, hc_attn_scores_avg, informed_metadata_bounds)

    # -- retrieving metadata regarding the UDPRS-Scale
    metadata_df = pd.read_csv(args.metadata_path, index_col=0)
    task_metadata_df = metadata_df[metadata_df['task_id'] == args.task_id]
    pd_metadata_df = task_metadata_df[task_metadata_df['sample_id'].isin(pd_sample_ids)].copy()

    # -- -- sorting the dataframe according to our data
    pd_metadata_df['sample_id_cat'] = pd.Categorical(
        pd_metadata_df['sample_id'],
        categories=pd_sample_ids,
        ordered=True,
    )
    pd_metadata_df.sort_values('sample_id_cat', inplace=True, ignore_index=True)

    # -- -- targeting a specific disease stadium
    updrs_speech_values = sorted(list(pd_metadata_df['updrs_speech'].unique()))

    # -- loading phoneme-level alignment
    aligned_frame_length = aligned_attn_scores.shape[1]
    phoneme_alignment = textgrids.TextGrid(args.phoneme_alignment)['phones']
    phoneme_alignment = phoneme_alignment_preprocessing(phoneme_alignment, study_coarticulations=args.study_coarticulations)
    print(phoneme_alignment)

    for phoneme_id, phoneme_segments in tqdm(phoneme_alignment.items(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        phoneme_data = []

        for updrs_speech_level in updrs_speech_values:
            target_idxs = np.where(pd_metadata_df['updrs_speech'] == updrs_speech_level)[0]

            phoneme_data += phoneme_analysis(pd_sample_ids, all_contrastive_attributions, target_idxs, phoneme_segments, variable=updrs_speech_level)

        # -- plotting per-phoneme analyses
        phoneme_data_df = pd.DataFrame(phoneme_data, columns=['sample_id', 'segment_id', 'UPDRS Scale', 'Informed High-Level Speech Dimension', f'Attention to phoneme /{phoneme_id}/'])

        # TODO -- statistical analysis
        # articulation_df = phoneme_data_df[phoneme_data_df['Informed High-Level Speech Dimension'] == r'$\bf{articulation}$']

        # articulation1 = articulation_df[articulation_df['UPDRS Scale'] == '1.0'][f'Attention to phoneme /{phoneme_id}/'].to_numpy()
        # articulation2 = articulation_df[articulation_df['UPDRS Scale'] == '2.0'][f'Attention to phoneme /{phoneme_id}/'].to_numpy()
        # articulation3 = articulation_df[articulation_df['UPDRS Scale'] == '3.0'][f'Attention to phoneme /{phoneme_id}/'].to_numpy()

        # statistic, pvalue = ks_2samp(articulation1, articulation3)
        # print(f'Significant difference between UPDRS 1.0 and UDPRS 3.0 for the phoneme {phoneme_id}: statistic={statistic} || p-value={pvalue}')

        # -- plotting data and saving stats
        if args.focused_study:
            for metadata_id in all_contrastive_attributions.keys():
                output_path = os.path.join(args.output_dir, metadata_id.replace('$\\bf{', '').replace('}$', ''), 'figures', f'phoneme_{phoneme_id}_analysis.png')
                focused_phoneme_data_df = phoneme_data_df[phoneme_data_df['Informed High-Level Speech Dimension'] == metadata_id]
                plotting_and_saving(focused_phoneme_data_df, phoneme_id, output_path, updrs_speech_values)

        else:
            output_path = os.path.join(args.output_dir, 'figures', f'phoneme_{phoneme_id}_analysis.png')
            plotting_and_saving(phoneme_data_df, phoneme_id, output_path, updrs_speech_values)

def plotting_and_saving(phoneme_data_df, phoneme_id, output_path, updrs_speech_values):
    plt.figure()

    sns_plot = sns.boxplot(data=phoneme_data_df, x='Informed High-Level Speech Dimension', y=f'Attention to phoneme /{phoneme_id}/', hue='UPDRS Scale', palette='tab10', boxprops={'alpha': 0.4})
    sns.stripplot(data=phoneme_data_df, x='Informed High-Level Speech Dimension', y=f'Attention to phoneme /{phoneme_id}/', hue='UPDRS Scale', dodge=True, palette='tab10', ax=sns_plot)

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

    # -- saving data to subsequent, potential analyses
    phoneme_data_df.to_csv(output_path.replace('/figures/', '/data/'), index=False)


if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Cross Temporal Attention Score Analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where model output can be retrieved.')
    parser.add_argument('--wav2vec-dir', required=True, type=str, help='Directory where wav2vec embeddings can be retrieved.')
    parser.add_argument('--metadata-path', required=True, type=str, help='Path to the CSV file containing the metadata of the addressed dataset.')
    parser.add_argument('--task-id', required=True, type=str, help='We thought was not necessary, but it is.')
    parser.add_argument('--filter-by-id', default='', type=str, help='In case you do not want the average across subjects, but only for a specific subset of specific sample')
    parser.add_argument('--phoneme-alignment', default='', type=str, help='TextGrid specifying the phoneme-level alignment for the minimum-length audio sample')
    parser.add_argument('--study-coarticulations', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--focused-study', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--output-dir', default='./plots/cross_full/phoneme_analysis/', type=str, help='Output directory where plots will be saved.')

    args = parser.parse_args()

    # -- main script execution
    main(args)
