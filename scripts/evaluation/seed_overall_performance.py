import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def get_reports(exps_dir):

    val_preds = []
    val_labels = []
    test_preds = []
    test_labels = []

    fold_dirs = os.listdir(exps_dir)
    for fold_dir in fold_dirs:
        model_output_dir = os.path.join(exps_dir, fold_dir, 'model_output')

        # -- validation set
        val_report_path = os.path.join(os.path.join(model_output_dir, 'validation_classification.pkl'))
        with open(val_report_path, 'rb') as f:
            val_model_output = pickle.load(f)

        val_preds += val_model_output['preds']
        val_labels += val_model_output['labels']

        # -- test set
        test_report_path = os.path.join(os.path.join(model_output_dir, 'test_classification.pkl'))
        with open(test_report_path, 'rb') as f:
            test_model_output = pickle.load(f)

        test_preds += test_model_output['preds']
        test_labels += test_model_output['labels']

    # -- computing reports
    val_report = classification_report(
        val_labels,
        val_preds,
        target_names=['HC', 'PD'],
        output_dict=False,
    )

    test_report = classification_report(
        test_labels,
        test_preds,
        target_names=['HC', 'PD'],
        output_dict=False,
    )

    return val_report, test_report

if __name__ == '__main__':

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Overall performance across the multiple assesing fold splits.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where the experiments are stored. It should include each directory for each run.')

    args = parser.parse_args()

    val_report, test_report = get_reports(args.exps_dir)

    print('', '-'*21, '\nVALIDATION SET\n', '-'*21, '\n', val_report)

    print('\n'*3, '-'*21, '\nTEST SET\n', '-'*21, '\n', test_report)
