import os
import sys
import copy
import torch
import pickle
import numpy as np
import pandas as pd

from os import path
sys.path.append( path.dirname(path.dirname(path.dirname( path.abspath(__file__)))) )

from models import *
from datasets import *

def build_model(config):
    if config.model == 'self_inf':
        model = SelfInfModel(config)
    elif config.model == 'self_ssl':
        model = SelfSSLModel(config)
    elif config.model == 'cross_full':
        model = CrossFullModel(config)
    elif config.model == 'RECAPD':
        model = RECAPD_Model(config)
    elif config.model == 'cross_full_fix':
        model = CrossFullModelFix(config)
    elif config.model == 'cross_full_fix_value':
        model = CrossFullModelFixValue(config)
    else:
        raise ValueError(f'unknown {config.model} model architecture')

    return model.to(dtype=getattr(torch, config.dtype), device=config.device)

def task_and_feature_filtering(old_config, filter_tasks, exclude_features, verbose=True):
    config = copy.copy(old_config)

    if 'ALL' not in filter_tasks:
        filtered_tasks = []
        consequent_features = []
        for task in config.tasks:
            if task['name'] in filter_tasks:
                filtered_tasks.append(task)
                consequent_features += task['features']
        config.tasks = filtered_tasks
        consequent_features = list(set(consequent_features))
    else:
        consequent_features = []
        for task in config.tasks:
            if task['name'] == 'SENTENCES':
                consequent_features += task['features']
            consequent_features += task['features']

    filtered_features = []
    for feature in config.features:
        if (feature['name'] in consequent_features) and (feature['name'] not in exclude_features):
            filtered_features.append(feature)
    config.features = filtered_features

    assert len(config.features) > 0, f'Ensure you specified the features you expected to use'
    assert len(config.tasks) > 0, f'Ensure you specified the tasks you expected to use'

    if verbose:
        print(f'Using the following features: {" | ".join([feature["name"].upper() for feature in config.features])}')
        print(f'Using the following tasks: {" | ".join([task["name"].upper() for task in config.tasks])}')

    return config

def get_dataloader(config, dataset_path, is_training=True):
    # -- creating dataset
    dataset = ParkinsonDataset(
        config,
        dataset_path,
        is_training=is_training,
    )

    # -- defining dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=is_training,
        batch_size=config.training_settings['batch_size'],
        num_workers=config.training_settings['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    return dataloader

def get_optimizer_and_scheduler(config, model, train_loader):
    # -- optimizer
    if config.training_settings['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            config.training_settings['learning_rate'] / 10,
        )
    else:
        raise ValueError(f'unknown {config.training_settings["optimizer"]} optimizer')

    # -- scheduler
    if config.training_settings['scheduler'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training_settings['learning_rate'],
            epochs=config.training_settings['epochs'],
            steps_per_epoch=len(train_loader),
            anneal_strategy=config.training_settings['anneal_strategy'],
        )
    else:
        raise ValueError(f'unknown {config.training_settings["scheduler"]} scheduler')

    return optimizer, scheduler

def save_checkpoint(model, output_dir, suffix):
    # -- creating output directories
    model_checkpoints_dir = os.path.join(output_dir, 'model_checkpoints')
    os.makedirs(model_checkpoints_dir, exist_ok=True)

    # -- saving model checkpoint
    model_checkpoint_path = os.path.join(model_checkpoints_dir, f'{suffix}.pth')
    print(f'Saving model checkpoint in {model_checkpoint_path}...')
    torch.save(model.state_dict(), model_checkpoint_path)

def save_embedding(embedding, output_dir, sample_id):
    # -- creating output directories
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    # -- saving model checkpoint
    embedding_path = os.path.join(embeddings_dir, f'{sample_id}.npz')
    np.savez_compressed(embedding_path, data=embedding.reshape(1, -1))

def save_model_output(output_stats, output_dir, suffix, save_attention_scores=True):
    # -- creating output directories
    model_outputs_dir = os.path.join(output_dir, 'model_output')
    os.makedirs(model_outputs_dir, exist_ok=True)
    print(f'Saving model output in {model_outputs_dir}...')

    # -- splitting output for efficiency issues
    mha_scores_output = {}
    for key in output_stats.copy().keys():
        if 'mha_scores' in key or 'metadata' in key:
            mha_scores_output[key] = output_stats.pop(key, None)
    mha_scores_output['labels'] = output_stats['labels'].copy()
    mha_scores_output['sample_id'] = output_stats['sample_id'].copy()

    # -- saving model output
    model_output_path = os.path.join(model_outputs_dir, f'{suffix}_classification.pkl')
    with open(model_output_path, 'wb') as handle:
        pickle.dump(output_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if save_attention_scores:
        model_output_path = os.path.join(model_outputs_dir, f'{suffix}_mha_scores.pkl')
        with open(model_output_path, 'wb') as handle:
            pickle.dump(mha_scores_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

def override_yaml(yaml_config, to_override):
    if to_override is not None:
        for new_setting in to_override:
            if new_setting.count(':') == 1:
                key, value = new_setting.split(':')
                value_type_func = type(yaml_config[key])
                if value_type_func == bool:
                    yaml_config[key] = value == 'true'
                else:
                    yaml_config[key] = value_type_func(value)

            elif new_setting.count(':') == 2:
                conf, key, value = new_setting.split(':')
                value_type_func = type(yaml_config[conf][key])
                if value_type_func == bool:
                    yaml_config[conf][key] = value == 'true'
                else:
                    yaml_config[conf][key] = value_type_func(value)

    return yaml_config
