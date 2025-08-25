import os
import glob
import argparse
from tqdm import tqdm

def audio_normalization():
    for wav_file in tqdm(wav_files):
        output_path = os.path.join(args.output_dir, os.path.basename(wav_file))
        os.system(f'ffmpeg-normalize {wav_file} -o {output_path} -ar 16000 -f')

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='Resample audio waveforms to 16kHz', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--wav-dir', type=str, default='./data/gita/audios/')
    parser.add_argument('--output-dir', required=True, type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(args.wav_dir, '*.wav'))

    audio_normalization()

