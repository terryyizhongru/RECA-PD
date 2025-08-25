import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def stratified_dataset_splitting(dataset_path, k_folds, seed=42):
    # -- reading entire dataset
    dataset_dir = os.path.dirname(dataset_path)
    dataset = pd.read_csv(dataset_path)

    # -- retrieving only essential information
    dataset = dataset[['subject_id', 'sample_id', 'task_id' , 'label', 'prosody', 'wav2vec', 'phonation', 'articulation', 'glottal']]

    sgkf_fulltrain_test = StratifiedGroupKFold(n_splits=k_folds, random_state=seed, shuffle=True)
    sgkf_train_validation = StratifiedGroupKFold(n_splits=k_folds, random_state=seed, shuffle=True)

    fulltrain_dataset_paths = []
    train_dataset_paths = []
    validation_dataset_paths = []
    test_dataset_paths = []

    stratify_fulltrain_test_based_on = dataset['label'].map(lambda x: str(x)) + '_' + dataset['task_id']
    for i, (fulltrain_idx, test_idx) in enumerate(sgkf_fulltrain_test.split(dataset, stratify_fulltrain_test_based_on, dataset.subject_id)):
        fulltrain = dataset.iloc[fulltrain_idx]

        stratify_train_validation_based_on = fulltrain['label'].map(lambda x: str(x)) + '_' + fulltrain['task_id']
        train_idx, validation_idx = next(sgkf_train_validation.split(fulltrain, stratify_train_validation_based_on, fulltrain.subject_id))

        train = fulltrain.iloc[train_idx]
        validation = fulltrain.iloc[validation_idx]
        test = dataset.iloc[test_idx]

        fold_dir = os.path.join(dataset_dir, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)

        fulltrain_dataset_paths.append(os.path.join(fold_dir, f'fulltrain.csv'))
        fulltrain.to_csv(fulltrain_dataset_paths[-1], index=False)

        train_dataset_paths.append(os.path.join(fold_dir, f'train.csv'))
        train.to_csv(train_dataset_paths[-1], index=False)

        validation_dataset_paths.append(os.path.join(fold_dir, f'validation.csv'))
        validation.to_csv(validation_dataset_paths[-1], index=False)

        test_dataset_paths.append(os.path.join(fold_dir, f'test.csv'))
        test.to_csv(test_dataset_paths[-1], index=False)

    return fulltrain_dataset_paths, train_dataset_paths, validation_dataset_paths, test_dataset_paths

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='Subject-independent stratified split of a dataset based on label and task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-path', type=str, default='./splits/gita/dataset.csv')
    parser.add_argument('--k-folds', type=int, default=5)
    args = parser.parse_args()

    stratified_dataset_splitting(args.dataset_path, args.k_folds)
