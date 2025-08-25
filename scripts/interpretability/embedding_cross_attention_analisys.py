import os
import pickle
import argparse
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from distutils.util import strtobool
from matplotlib.colors import LogNorm
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler

plt.rcParams['text.usetex'] = True

def compute_cross_attn_embedding_analysis(attn_scores_avg, output_path):

    # -- thresholding
    # thr = np.quantile(attn_scores_avg, 0.95)
    # attn_scores_avg[attn_scores_avg < thr] = 1e-10

    # -- saving heatmap plot
    plt.figure()
    avg_heatmap_plot = sns.heatmap(attn_scores_avg, norm=LogNorm())
    avg_heatmap_plot.get_figure().savefig(output_path)

def compute_feature_relevance_analysis(attn_scores_avg, informed_metadata, output_path, top_features=10):

    # -- accumulating relevance per informed feature
    accum_attn_scores = attn_scores_avg.sum(axis=0)

    # -- informed speech metadata
    informed_metadata_ids, informed_metadata_bounds = informed_metadata

    analysis = []
    for current_idx, accum_score in enumerate(accum_attn_scores):
        informed_highlevel_group = get_informed_highlevel_group(current_idx, informed_metadata_bounds)
        metadata_id = informed_metadata_ids[current_idx]
        analysis.append( (accum_score, informed_highlevel_group, metadata_id) )

    top_analysis = sorted(analysis, key=lambda x: x[0], reverse=True)[:top_features]

    # -- preparing matrices for heatmap plot
    scores = np.array([[score] for score, _, _ in top_analysis])
    labels = np.array([[f'{metadata_id} (' + r'$\bf{' + type_inf + '}$)'] for _, type_inf, metadata_id in top_analysis])

    # -- saving heatmap plot
    plt.figure()
    avg_heatmap_plot = sns.heatmap(scores, annot=labels, norm=LogNorm(), cmap='Blues', fmt='')
    avg_heatmap_plot.get_figure().savefig(output_path)

def compute_highlevel_embedding_feature_attribution_analysis(attn_scores_avg, informed_metadata, output_path):

    # -- relevance thresholding
    thr = np.quantile(attn_scores_avg, 0.95)
    attn_scores_avg[attn_scores_avg < thr] = 1e-10

    # -- accumulating relevance per informed feature
    accum_attn_scores = attn_scores_avg.sum(axis=0)

    # -- informed speech metadata
    _, informed_metadata_bounds = informed_metadata

    # -- relevance by high-level speech information
    barplot_data = {'informed_ids': [], 'attributions': []}
    for metadata_id, (left_bound, right_bound) in informed_metadata_bounds.items():
        informed_attn_scores = accum_attn_scores[left_bound:right_bound]
        norm_informed_attn_scores = informed_attn_scores.mean()
        # norm_informed_attn_scores = informed_attn_scores.sum() / accum_attn_scores.sum()

        barplot_data['informed_ids'].append(r'$\bf{'+metadata_id+'}$')
        barplot_data['attributions'].append(norm_informed_attn_scores)

    barplot_data['attributions'] = np.array(barplot_data['attributions'])
    barplot_data['attributions'] = barplot_data['attributions'] / barplot_data['attributions'].sum()
    barplot_data['attributions'] = (barplot_data['attributions'] * 100.).tolist()

    # -- saving barplot
    plt.figure()
    plt.ylim(0, 100)
    avg_barplot = sns.barplot(x=barplot_data['informed_ids'], y=barplot_data['attributions'])
    avg_barplot.get_figure().savefig(output_path)

    return barplot_data

def create_lowlevel_polarplot(attn_scores, informed_metadata, output_path):

    # -- informed speech metadata
    informed_metadata_ids, informed_metadata_bounds = informed_metadata

    data = {
        'polarplot': {},
        # 'informed_ids': [('articulation', 'F1'), ('articulation', 'F2'), ('glottal', 'GCI'), ('glottal', 'NAQ'), ('glottal', 'QOQ'), ('glottal', 'HRF'), ('phonation', 'average DF0'), ('phonation', 'Jitter'), ('phonation', 'Shimmer'), ('phonation', 'APQ'), ('phonation', 'PPQ'), ('phonation', 'logE'), ('phonation', 'std DF0'), ('prosody', 'average F0'), ('prosody', 'std F0'), ('prosody', 'Evoiced'), ('prosody', 'Vrate'), ('prosody', 'durpause'), ('prosody', 'PVU'), ('prosody', 'UVU'), ('prosody', 'VVU')],
        'informed_ids': [
            ('articulation', 'average F1'), ('articulation', 'std F1'), ('articulation', 'average F2'), ('articulation', 'std F2'),
            ('glottal', 'global average var GCI'), ('glottal', 'global average avg NAQ'), ('glottal', 'global average std NAQ'), ('glottal', 'global average avg QOQ'), ('glottal', 'global average std QOQ'), ('glottal', 'global average avg HRF'), ('glottal', 'global average std HRF'), ('glottal', 'global std var GCI'), ('glottal', 'global std avg NAQ'), ('glottal', 'global std std NAQ'), ('glottal', 'global std avg QOQ'), ('glottal', 'global std std QOQ'), ('glottal', 'global std avg HRF'), ('glottal', 'global std std HRF'),
            ('phonation', 'average DF0'), ('phonation', 'average Jitter'), ('phonation', 'average Shimmer'), ('phonation', 'average APQ'), ('phonation', 'average PPQ'), ('phonation', 'average logE'), ('phonation', 'std DF0'),
            ('prosody', 'average F0'), ('prosody', 'std F0'), ('prosody', 'average Evoiced'), ('prosody', 'std Evoiced'), ('prosody', 'Vrate'), ('prosody', 'average durpause'), ('prosody', 'std durpause'), ('prosody', 'PVU'), ('prosody', 'UVU'), ('prosody', 'VVU')],
    }

    # scaler = MinMaxScaler(feature_range=(0,1))
    for condition_id in attn_scores.keys():
        data['polarplot'][condition_id] = {'attributions': []}
        accum_attn_scores = attn_scores[condition_id].sum(axis=0)

        for highlevel_group, informed_id in data['informed_ids']:
            accum_stats = []

            for current_idx, accum_score in enumerate(accum_attn_scores):
                current_informed_id = informed_metadata_ids[current_idx]
                current_highlevel_group = get_informed_highlevel_group(current_idx, informed_metadata_bounds)

                if highlevel_group == current_highlevel_group:
                    if informed_id in current_informed_id:
                        accum_stats.append( accum_score )

            assert len(accum_stats) > 0, f'we did not gather any stat for {highlevel_group} -- {informed_id}?'
            accum_stats = np.array(accum_stats).sum()
            data['polarplot'][condition_id]['attributions'].append( accum_stats )

        # data['polarplot'][condition_id]['attributions'] = scaler.fit_transform( np.array(data['polarplot'][condition_id]['attributions']).reshape(-1,1) )
        # data['polarplot'][condition_id]['attributions'] = (np.squeeze(data['polarplot'][condition_id]['attributions']) * 100.0).tolist()
        data['polarplot'][condition_id]['attributions'] = np.array(data['polarplot'][condition_id]['attributions'])
        data['polarplot'][condition_id]['attributions'] = (data['polarplot'][condition_id]['attributions'] / data['polarplot'][condition_id]['attributions'].sum()) * 100.0
        data['polarplot'][condition_id]['attributions'] = data['polarplot'][condition_id]['attributions'].tolist()

        highlevel_groups = [highlevel_group for highlevel_group, _ in data['informed_ids']]
        data['polarplot'][condition_id]['informed_ids'] = [metadata_id for _, metadata_id in data['informed_ids']]

    with open(output_path.replace('figures', 'stats').replace('.png', '.pkl'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    create_polarplot(data['polarplot'], output_path, lowlevel=True, highlevel_groups=highlevel_groups)


def create_polarplot(polarplot_data, output_path, lowlevel=False, highlevel_groups=None):
    first_key = list(polarplot_data.keys())[0]

    # -- saving polar plot
    plt.figure()
    N = len(polarplot_data[first_key]['informed_ids'])

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(1,1,1, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], polarplot_data[first_key]['informed_ids'], color='black', size=12)
    ax.tick_params(axis='x', which='major', pad=21)

    ax.set_rlabel_position(0)
    if args.study_lowlevel_by_condition:
        plt.yticks([20,40,60,80,100], ['20%', '40%', '60%', '80%', '100%'], color='black', size=10)
        plt.ylim(0,50)
    else:
        plt.yticks([-10, -5, 0, 5, 10, 15], ['','','','5%','%10','%15'], color='black', size=10)
        plt.ylim(0,20)

    if lowlevel:
        rlabels = ax.get_xmajorticklabels()
        colors = {'articulation': '#1f77b4', 'glottal': '#ff7f0e', 'phonation': '#2ca02c', 'prosody': '#d62728'}
        for highlevel_group, label in zip(highlevel_groups, rlabels):
            label.set_color( colors[highlevel_group]  )

        new_rlabels = [label.get_text().replace('average', 'avg') for label in rlabels]
        ax.set_xticklabels(new_rlabels)

    ax.set_xticklabels([])
    colors = ['#81cfe0', '#6a89cc']
    for i, condition in enumerate(polarplot_data):
        values = polarplot_data[condition]['attributions']
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=1.5, marker='o', linestyle='solid', label=condition)
        ax.fill(angles, values, color=colors[i], alpha=0.2)

    lgd = ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(.5, 1.15), title=r'$\bf{Condition}$')

    if lowlevel:
        blue_patch = mpatches.Patch(color='#1f77b4', label='articulation')
        orange_patch = mpatches.Patch(color='#ff7f0e', label='glottal')
        green_patch = mpatches.Patch(color='#2ca02c', label='phonation')
        red_patch = mpatches.Patch(color='#d62728', label='prosody')

        # blue_patch = Line2D([0], [0], color='#1f77b4', lw=1.5, label='articulation')
        # orange_patch = Line2D([0], [0], color='#ff7f0e', lw=1.5, label='glottal')
        # green_patch = Line2D([0], [0], color='#2ca02c', lw=1.5, label='phonation')
        # red_patch = Line2D([0], [0], color='#d62728', lw=1.5, label='prosody')

        lgd2 = plt.legend(handles=[blue_patch, orange_patch, green_patch, red_patch], loc='upper center', bbox_to_anchor=(.5, -0.15), ncol=4, title=r'$\bf{High-Level~Speech~Dimensions}$')
        plt.gca().add_artist(lgd)
        plt.savefig(output_path, bbox_extra_artists=(lgd, lgd2), bbox_inches='tight')

    else:
        plt.savefig(output_path, bbox_extra_artists=(lgd,), bbox_inches='tight')

def contrast_attn_score_matrices(attn_scores, output_path):

    # -- relevance thresholding
    for condition_id in attn_scores.keys():
        thr = np.quantile(attn_scores[condition_id], 0.95)
        attn_scores[condition_id][attn_scores[condition_id] < thr] = 1e-10

    # -- saving heatmap plot
    plt.figure()
    plt.tight_layout()
    plt.gca().set_aspect('equal')

    # -- presence and exclusive masks
    hc_mask = attn_scores['HC'] > 1e-10
    pd_mask = attn_scores['PD'] > 1e-10
    xor_mask = np.logical_xor(hc_mask, pd_mask)

    # -- difference template plot
    diff_template = np.zeros(xor_mask.shape)

    diff_template[np.where(hc_mask)] = 1
    diff_template[np.where(pd_mask)] = -1
    diff_template[np.where(~xor_mask)] = 0

    # -- custom discrete color map
    cmap = colors.ListedColormap(['#ffa500', '#03051a', '#05bbaa'])
    bounds=[-1, -0.5, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # -- add discrete legend
    orange_patch = mpatches.Patch(color='#ffa500', label='PD')
    blue_patch = mpatches.Patch(color='#05bbaa', label='HC')
    plt.legend(handles=[orange_patch, blue_patch], loc='center left', bbox_to_anchor=(1, 0.5))

    # -- axis tick labels
    num_inf_feats = attn_scores['HC'].shape[-1]
    xticks = list(range(0, num_inf_feats, 24 if num_inf_feats > 64 else 3))
    plt.xticks(xticks, xticks, rotation=90)
    plt.xlabel(r'$\bf{'+'Informed~Speech~Features'+'}$')

    num_ssl_feats = attn_scores['HC'].shape[0]
    yticks = list(range(0, num_ssl_feats, 40))
    plt.yticks(yticks, yticks[::-1])
    plt.ylabel(r'$\bf{'+'Wav2Vec~Embedding~Dimension'+'}$')

    # -- actual plotting
    aspect_ratio = num_inf_feats / num_ssl_feats
    plt.imshow(diff_template, aspect=aspect_ratio,
        interpolation='nearest', origin='lower',
        cmap=cmap, norm=norm,
    )

    plt.savefig(output_path, format='pdf')

def main(args, filter_by_condition=0, output_filename='HC_SUBJECTS'):

    # -- loading target model outputs and informed speech features metadata
    _, attn_scores, informed_metadata = load_model_output(
        args.exps_dir,
        args.filter_by_id,
        filter_by_condition,
        attn_scores_id='cross_embed_mha_scores',
    )

    attn_scores = np.array(attn_scores)
    attn_scores_avg = attn_scores.mean(axis=0).copy()

    print(f'\nConsidering {attn_scores.shape[0]} {"HC" if filter_by_condition == 0 else "PD"} samples based on the ID filter: {args.filter_by_id}')

    # -- cross attention heatmap analysis
    output_path = os.path.join(args.output_dir, 'figures', f'{output_filename}_embedding_attn_scores.png')
    compute_cross_attn_embedding_analysis(attn_scores_avg, output_path)

    # -- feature relevance heatmap analysis
    output_path = os.path.join(args.output_dir, 'figures', f'{output_filename}_embedding_top{args.top_features}_feature_relevance.png')
    compute_feature_relevance_analysis(attn_scores_avg, informed_metadata, output_path, top_features=args.top_features)

    # -- high-level feature attribution analysis
    output_path = os.path.join(args.output_dir, 'figures', f'{output_filename}_embedding_highlevel_attribution.png')
    highlevel_attributions = compute_highlevel_embedding_feature_attribution_analysis(attn_scores_avg, informed_metadata, output_path)

    return attn_scores, attn_scores_avg, highlevel_attributions, informed_metadata

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Cross Embedding Attention Score Analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where model output can be retrieved.')
    parser.add_argument('--top-features', default=10, type=int, help='Number of top features you want to analyze')
    parser.add_argument('--filter-by-id', default='', type=str, help='In case you do not want the average across subjects, but only for a specific subset of specific sample')
    parser.add_argument('--study-lowlevel-by-condition', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--output-dir', default='./plots/cross_ssl_full/embedding/', type=str, help='Output directory where plots will be saved.')

    args = parser.parse_args()

    # -- creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['figures', 'stats']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # -- main script execution
    hc_cross_attn_scores, hc_cross_attn_scores_avg, hc_highlevel_attributions, _ = main(args, filter_by_condition=0, output_filename='HC_SUBJECTS')
    pd_cross_attn_scores, pd_cross_attn_scores_avg, pd_highlevel_attributions, informed_metadata = main(args, filter_by_condition=1, output_filename='PD_SUBJECTS')

    # -- polarplot for highlevel informed feature attribution
    polarplot_data = {'HC': hc_highlevel_attributions, 'PD': pd_highlevel_attributions}
    output_path = os.path.join(args.output_dir, 'figures', f'highlevel_attributions_polarplot.png')
    create_polarplot(polarplot_data, output_path)

    # -- fusing attention matrix scores
    attn_scores = {'HC': hc_cross_attn_scores_avg, 'PD': pd_cross_attn_scores_avg}
    output_path = os.path.join(args.output_dir, 'figures', f'contrast_attn_scores.pdf')
    contrast_attn_score_matrices(attn_scores, output_path)

    # -- polarplot for lowlevel informed feature attribution
    output_path = os.path.join(args.output_dir, 'figures', f'lowlevel_attributions_polarplot.png')
    if args.study_lowlevel_by_condition:
        create_lowlevel_polarplot(attn_scores, informed_metadata, output_path)
    else:
        cross_attn_scores_avg = np.vstack( (hc_cross_attn_scores, pd_cross_attn_scores) ).mean(axis=0)
        create_lowlevel_polarplot({'ALL': cross_attn_scores_avg}, informed_metadata, output_path)
