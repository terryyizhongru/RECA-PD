import os
import glob
import argparse
import numpy as np
from utils import *
import pandas as pd
import seaborn as sns
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm

def compute_padding_average(attn_scores):

    # -- applying sequence temporal padding
    max_length = max([sample.shape[0] for sample in attn_scores])
    attn_scores = [np.pad(sample, ((0,max_length-sample.shape[0]),(0,0)), mode='constant', constant_values=np.nan) for sample in attn_scores]
    attn_scores_np = np.array(attn_scores)

    # -- computing average, while ignoring NaNs items
    attn_scores_avg = np.nanmean(attn_scores_np, axis=0)
    attn_scores_avg[np.isnan(attn_scores_avg)] = 1e-10

    return attn_scores_avg

def get_min_length_embedding(embeddings):
    min_idx = 0
    min_len = 1e10

    for idx, embedding in enumerate(embeddings):
        emb_len = embedding.shape[0]
        if emb_len < min_len:
            min_idx = idx
            min_len = emb_len

    return min_idx

def align_attn_score(path, attn_score):
    aligned_attn_score = np.empty((0, attn_score.shape[-1]))

    start_idx = 0
    end_idx = start_idx

    for i, (target_idx, ref_idx) in enumerate(path):

        if i + 1 < len(path):
            next_ref_idx = path[i+1][-1]
        else:
            # -- we reach the end of the series, so we collapse
            next_ref_idx = None

        # -- collapsing either 1 or more timesteps
        if ref_idx != next_ref_idx:
            end_idx = target_idx + 1

            # -- special case when the shorter series should be further shrinked
            # -- however in this case we just apply repeatitions to about that shrinking
            if end_idx - start_idx == 0:
                start_idx = start_idx - 1

            attn_scores_to_stack = attn_score[start_idx:end_idx, :].mean(axis=0)

            # -- ongoing alignment
            aligned_attn_score = np.vstack((
                aligned_attn_score,
                attn_scores_to_stack,
            ))

            # -- update the start index
            start_idx = end_idx

    return aligned_attn_score

def apply_dtw_alignment(wav2vec_dir, sample_ids, attn_scores):
    embeddings = []
    for sample_id in sample_ids:
        sample_paths = glob.glob(f'{wav2vec_dir}{os.path.sep}*{sample_id}*')
        assert len(sample_paths) == 1, f"Isn't the sample ID unique?: {sample_paths}"

        embeddings.append( np.load(sample_paths[0])['data'] )

    aligned_attn_scores = []
    ref_idx = get_min_length_embedding(embeddings)

    for idx, (embedding, attn_score) in enumerate(zip(embeddings, attn_scores)):
        if idx == ref_idx:
            aligned_attn_scores.append( attn_score )
        else:
            path, dist = dtw_path(embedding, embeddings[ref_idx])
            aligned_attn_scores.append( align_attn_score(path, attn_score)  )

    return np.array(aligned_attn_scores)

def compute_cross_attn_temporal_analysis(attn_scores, output_path):

    # -- computing average across attention weights
    attn_scores_avg = attn_scores.mean(axis=0)

    # -- aplying relevance threshold for visualization
    thr = np.quantile(attn_scores_avg, 0.95)
    attn_scores_avg[attn_scores_avg < thr] = 1e-10

    # -- saving heatmap analysis plot
    plt.figure()
    avg_heatmap_plot = sns.heatmap(attn_scores_avg.T, norm=LogNorm())
    avg_heatmap_plot.get_figure().savefig(output_path)

    return attn_scores_avg

def compute_highlevel_temporal_feature_attribution_analysis(attn_scores_avg, informed_metadata, output_path, sample_rate=49):

    # -- aplying relevance threshold for visualization
    # # thr = np.quantile(attn_scores_avg, 0.95, axis=(1,-1))
    # # attn_scores_avg[attn_scores_avg < thr[:, None, None]] = 1e-10

    # -- retrieving informed speech feature metadata information
    _, informed_metadata_bounds = informed_metadata

    # -- time-wise accumulative relevance per high-level feature type
    lineplot_mean_data = {}
    total_mean_attribution = np.zeros(attn_scores_avg.shape[0])
    # # total_attribution = np.zeros((attn_scores_avg.shape[0], attn_scores_avg.shape[1]))
    for metadata_id, (left_bound, right_bound) in informed_metadata_bounds.items():
        informed_attn_scores = attn_scores_avg[:, left_bound:right_bound]
        # # informed_attn_scores = attn_scores_avg[:, :, left_bound:right_bound]

        attribution_mean_scores = informed_attn_scores.mean(axis=-1)
        lineplot_mean_data[r'$\bf{'+metadata_id+'}$'] = attribution_mean_scores

        total_mean_attribution += attribution_mean_scores

    # -- normalizing attribution scores
    for metadata_id, _ in informed_metadata_bounds.items():
        lineplot_mean_data[r'$\bf{'+metadata_id+'}$'] = (lineplot_mean_data[r'$\bf{'+metadata_id+'}$'] / total_mean_attribution) * 100.

    # -- converting time to seconds, taking into account Wav2Vec processes speech at 49Hz
    lineplot_mean_data['Time (seconds)'] = np.arange(attribution_mean_scores.shape[0]) / sample_rate

    # -- saving barplot
    plt.figure()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6.0), gridspec_kw={'width_ratios': [3, 1]})

    ax[0].grid()
    ax[0].set_ylim(-1, 100)

    data_mean_df = pd.DataFrame(lineplot_mean_data)
    data_mean_df = pd.melt(data_mean_df, id_vars=['Time (seconds)'])
    data_mean_df = data_mean_df.rename(columns={'value': 'Relevance Score'})

    # # data_df = pd.DataFrame({k:x.ravel() for k,x in lineplot_data.items()}, index=pd.MultiIndex.from_product([np.arange(attribution_scores.shape[0]), np.arange(attribution_scores.shape[1])], names=['Sample', 'Time (seconds)'])).reset_index()
    # # data_df['Time (seconds)'] = data_df['Time (seconds)'] / sample_rate
    # # data_df = pd.melt(data_df, ['Sample','Time (seconds)'])

    time_lineplot = sns.lineplot(ax=ax[0], data=data_mean_df, x='Time (seconds)', y='Relevance Score', hue='variable', palette="tab10", linewidth=1.5)

    for line in time_lineplot.lines:
        x, y = line.get_xydata().T
        time_lineplot.fill_between(x, 0, y, color=line.get_color(), alpha=.33)

    time_lineplot.get_legend().set_title(None)

    data_mean_df = data_mean_df.drop('Time (seconds)', axis=1)
    data_total_df = data_mean_df.groupby('variable').sum()

    total_barplot = sns.barplot(ax=ax[1], x=data_total_df.index, y=data_total_df['Relevance Score'])
    total_barplot.set_xticklabels(rotation=30, labels=data_total_df.index)

    time_lineplot.figure.savefig(output_path)

    return lineplot_mean_data

def main(args, filter_by_condition=0, output_filename='HC_SUBJECTS'):

    # -- loading target model outputs and informed speech features metadata
    sample_ids, attn_scores, informed_metadata = load_model_output(
        args.exps_dir,
        args.filter_by_id,
        filter_by_condition,
        attn_scores_id='cross_time_mha_scores',
        return_df=True,
    )

    print(f'\nConsidering {len(attn_scores)} {"HC" if filter_by_condition == 0 else "PD"} samples based on the ID filter: {args.filter_by_id}')

    # -- applying dynamic time warping
    aligned_attn_scores = apply_dtw_alignment(args.wav2vec_dir, sample_ids, attn_scores)

    # -- cross attention heatmap analysis
    output_path = os.path.join(args.output_dir, 'figures', f'{output_filename}_temporal_attn_scores.png')
    attn_scores_avg = compute_cross_attn_temporal_analysis(aligned_attn_scores, output_path)

    # -- high-level feature attribution analysis
    output_path = os.path.join(args.output_dir, 'figures', f'{output_filename}_temporal_highlevel_attribution.png')
    highlevel_attributions = compute_highlevel_temporal_feature_attribution_analysis(attn_scores_avg, informed_metadata, output_path)

    # -- saving statistics
    output_path = os.path.join(args.output_dir, 'stats', f'{output_filename}_temporal_attn_scores.npz')
    np.savez_compressed(output_path, data=attn_scores_avg)

    output_path = os.path.join(args.output_dir, 'stats', f'{output_filename}_temporal_highlevel_attribution.npz')
    np.savez_compressed(output_path, data=highlevel_attributions)

    return attn_scores_avg, highlevel_attributions

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Cross Temporal Attention Score Analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where model output can be retrieved.')
    parser.add_argument('--wav2vec-dir', required=True, type=str, help='Directory where wav2vec embeddings can be retrieved.')
    parser.add_argument('--filter-by-id', default='', type=str, help='In case you do not want the average across subjects, but only for a specific subset of specific sample')
    # parser.add_argument('--use-dtw', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--output-dir', default='./plots/cross_ssl_full/embedding/', type=str, help='Output directory where plots will be saved.')

    args = parser.parse_args()

    # -- creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['figures', 'stats']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # -- main script execution
    hc_cross_attn_scores_avg, hc_highlevel_attributions = main(args, filter_by_condition=0, output_filename='HC_SUBJECTS')
    pd_cross_attn_scores_avg, pd_highlevel_attributions = main(args, filter_by_condition=1, output_filename='PD_SUBJECTS')
