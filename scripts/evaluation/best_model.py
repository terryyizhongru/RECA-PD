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


    best_f1 = 0.0
    val_model_output_paths = glob.glob(f'{args.exps_dir}/**/model_output/validation_classification.pkl', recursive=True)

    for val_model_output_path in val_model_output_paths:
        run_id = val_model_output_path.split('/')[-4]
        fold_id = val_model_output_path.split('/')[-3]

        # -- validation model output
        with open(val_model_output_path, 'rb') as f:
            val_model_output = pickle.load(f)

        val_preds = val_model_output['preds']
        val_labels = val_model_output['labels']

        # -- computing classification report
        val_report = classification_report(
            val_labels,
            val_preds,
            target_names=['HC', 'PD'],
            output_dict=True,
        )

        val_f1 = val_report['macro avg']['f1-score']
        # print(val_f1, run_id, fold_id)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_run_id = run_id
            best_fold_id = fold_id

    # BEST PERFORMING MODEL
    best_model_output_dir = os.path.join(args.exps_dir, best_run_id, best_fold_id, 'model_output')
    best_model_output_val_path = os.path.join(best_model_output_dir, 'validation_classification.pkl')
    best_model_output_test_path = os.path.join(best_model_output_dir, 'test_classification.pkl')
    print(f'\nBEST PERFORMING MODEL: {best_model_output_dir}\n')

    # -- validation set
    with open(best_model_output_val_path, 'rb') as f:
        best_model_output_val = pickle.load(f)

    best_val_preds = best_model_output_val['preds']
    best_val_labels = best_model_output_val['labels']

    # -- computing classification report
    val_report = classification_report(
        best_val_labels,
        best_val_preds,
        target_names=['HC', 'PD'],
    )
    print(val_report)

    # -- test set

    with open(best_model_output_test_path, 'rb') as f:
        best_model_output_test = pickle.load(f)

    best_test_preds = best_model_output_test['preds']
    best_test_labels = best_model_output_test['labels']

    # -- computing classification report
    test_report = classification_report(
        best_test_labels,
        best_test_preds,
        target_names=['HC', 'PD'],
    )
    print(test_report)

            
