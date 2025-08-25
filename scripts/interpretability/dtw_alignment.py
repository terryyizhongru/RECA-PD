import os
import glob
import numpy as np
import seaborn as sns
from tslearn.metrics import dtw_path

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

def apply_dtw_alignment(wav2vec_dir, task_id, sample_ids, attn_scores):
    embeddings = []
    for sample_id in sample_ids:
        sample_paths = glob.glob(f'{wav2vec_dir}{os.path.sep}*_{task_id}_*{sample_id}*')
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
