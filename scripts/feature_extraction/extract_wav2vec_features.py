import os
import glob
import torch
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='WAV2VEC-based Feature Embedding Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wav-dir', type=str, default='./data/gita/norm_audios/')
    parser.add_argument('--chunk-size', type=int, default=30)
    parser.add_argument('--output-dir', required=True, type=str)
    args = parser.parse_args()

    # -- building wav2vec model
    nlayers = [7]
    for i in nlayers:
        feature_output_dir = os.path.join(args.output_dir, 'wav2vec', f'layer{str(i).zfill(2)}')
        os.makedirs(feature_output_dir, exist_ok=True)

    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    model = bundle.get_model()

    wav_paths = sorted(glob.glob(f'{args.wav_dir}{os.sep}*.wav'))

    # for wav_path in tqdm(wav_paths):
    for wav_path in tqdm(wav_paths):
        sample_id = os.path.basename(wav_path).replace('.wav', '.npz')

        waveform, sr = torchaudio.load(wav_path)
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        # -- in case the sample is too long
        waveform_seconds = waveform.shape[-1] / sr
        if waveform_seconds > 60:
            nchunks = int(waveform_seconds / args.chunk_size)

            for layer_id in nlayers:
                feature = torch.empty((0,1024))

                for waveform_chunk in waveform.chunk(nchunks, dim=1):
                    chunk_features = model.extract_features(waveform_chunk)[0][layer_id]
                    feature = torch.vstack((feature, chunk_features.squeeze(0)))

                output_path = os.path.join(args.output_dir, 'wav2vec', f'layer{str(layer_id).zfill(2)}', sample_id)
                feature_to_save = feature.cpu().detach().numpy()
                # print(output_path, feature_to_save.shape)
                np.savez_compressed(output_path, data=feature_to_save)
        else:
            for layer_id in nlayers:
                features = model.extract_features(waveform)[0][layer_id]

                output_path = os.path.join(args.output_dir, 'wav2vec', f'layer{str(layer_id).zfill(2)}', sample_id)
                feature_to_save = features.squeeze(0).cpu().detach().numpy()
                # print(output_path, feature_to_save.shape)
                np.savez_compressed(output_path, data=feature_to_save)
