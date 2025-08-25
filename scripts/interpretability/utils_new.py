import glob
import numpy as np
import pickle as pkl

def read_pickle(pickle_path):
    # -- loading model output pickle
    with open(pickle_path, "rb") as f:
        data = pkl.load(f)

    return data

def load_model_output(exps_dir, filter_by_id, filter_by_condition, attn_scores_id='cross_embed_mha_scores'):
    # -- retrieving test model output paths
    model_cls_output_paths = glob.glob(f'{exps_dir}/**/model_output/test_classification.pkl', recursive=True)
    model_mha_output_paths = glob.glob(f'{exps_dir}/**/model_output/test_mha_scores.pkl', recursive=True)

    print(f'Found {len(model_cls_output_paths)} model classification output files')
    print(f'Found {len(model_mha_output_paths)} model mha output files')
    # -- loading model output
    sample_ids = []
    target_attn_scores = []
    for model_cls_output_path in model_cls_output_paths:
        print(f'Loading model classification output from: {model_cls_output_path}')
        batch_cls_model_outputs = read_pickle(model_cls_output_path)
        
        print(model_cls_output_path)
        for s, pred, label, sample_id, attn_scores in zip(
            batch_cls_model_outputs['sample_id'],
            batch_cls_model_outputs['preds'],

            batch_cls_model_outputs[attn_scores_id],
        ):
            print(f'Processing sample: {sample_id}, label: {label}, pred: {pred}')
            print(f'  Attn scores shape: {attn_scores.shape}')
            print(batch_cls_model_outputs.keys())
            # -- filtering by subset, or specific sample
            if filter_by_id in sample_id:
                if filter_by_condition == label:
                    if pred == label:
                        sample_ids.append( sample_id )
                        target_attn_scores.append( np.squeeze(attn_scores, axis=0) )

    assert len(target_attn_scores) > 0, f'\nNo ID filter matching found for: {filter_by_id}'

    # -- getting metadata regarding informed speech feaures IDs
    informed_metadata = batch_mha_model_outputs['informed_metadata']
    return sample_ids, target_attn_scores, informed_metadata

def get_informed_highlevel_group(current_idx, informed_metadata_bounds):
    for type_inf, (start, end) in informed_metadata_bounds.items():
        if current_idx >= start and current_idx < end:
            return type_inf
