import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Overall performance across the multiple assesing fold splits.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where the experiments are stored. It should include each directory for each run.')

    args = parser.parse_args()

    # -- get run run IDs
    run_ids = os.listdir(args.exps_dir)

    best_run_id = ''
    best_run_f1 = 0.0
    for run_id in run_ids:
        run_dir = os.path.join(args.exps_dir, run_id)

        val_model_output_paths = glob.glob(f'{run_dir}/**/model_output/validation_classification.pkl', recursive=True)

        run_preds = []
        run_labels = []
        for val_model_output_path in val_model_output_paths:
            # -- validation model output
            with open(val_model_output_path, 'rb') as f:
                val_model_output = pickle.load(f)

            run_preds += val_model_output['preds']
            run_labels += val_model_output['labels']

        # -- computing classification report
        run_report = classification_report(
            run_labels,
            run_preds,
            target_names=['HC', 'PD'],
            output_dict=True,
        )

        run_f1 = run_report['macro avg']['f1-score']

        # -- find out the better run run ID
        if run_f1 > best_run_f1:
            best_run_f1 = run_f1
            best_run_id = run_id

    # BEST PERFORMING MODEL
    print(f'The best-performing run ID model is: {os.path.join(args.exps_dir, best_run_id)} with an average F1-score of {best_run_f1}.')
