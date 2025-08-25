import sys
import yaml
import torch
import random
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
from pathlib import Path
from colorama import Fore
from distutils.util import strtobool
from sklearn.metrics import accuracy_score, classification_report

def train(config):
    model.train()
    optimizer.zero_grad()

    train_output = {'loss': 0.0, 'acc': 0.0}
    accum_grad = config.training_settings['accum_grad']
    for batch_idx, batch in enumerate(tqdm(train_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.GREEN, Fore.RESET))):
        batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

        # -- forward pass
        model_output = model(batch)
        loss = model_output['loss'] / accum_grad

        # -- optimization
        loss.backward()
        if ((batch_idx+1) % accum_grad == 0) or (batch_idx+1 == len(train_loader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_output['loss'] += loss.item()
        train_output['acc'] += accuracy_score(
            model_output['preds'].detach().cpu().numpy(),
            model_output['labels'].detach().cpu().numpy()
        )

    train_output['loss'] = train_output['loss'] / (len(train_loader) / accum_grad)
    train_output['acc'] = (train_output['acc'] / len(train_loader)) * 100.0

    return train_output

def evaluate(config, eval_loader, is_test=False):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, position=0, leave=True, file=sys.stdout, bar_format="{l_bar}%s{bar:10}%s{r_bar}" % (Fore.BLUE, Fore.RESET))):
            batch = {k: v.to(device=config.device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}

            # -- forward pass
            model_output = model(batch)

            # -- evaluation output initialization
            if batch_idx == 0:
                eval_output = {model_output_key:[] for model_output_key in list(model_output.keys())+['acc']}

            # -- gathering statistics
            for eval_key in eval_output.keys():
                if eval_key == 'loss':
                    eval_output[eval_key] += [model_output['loss'].item()]
                elif eval_key == 'acc':
                    eval_output['acc'] += [accuracy_score(
                        model_output['preds'].detach().cpu().numpy(),
                        model_output['labels'].detach().cpu().numpy(),
                    )]
                elif eval_key == 'cross_time_mha_scores':
                    time_mha_scores = []
                    for i, time_mha_score_sample in enumerate(model_output[eval_key].detach().cpu().numpy()):
                        time_mha_scores.append( time_mha_score_sample[:, :batch['ssl_lengths'][i], :] )
                    eval_output[eval_key] += time_mha_scores
                elif eval_key in ['self_inf_mha_scores', 'self_ssl_mha_scores', 'cross_embed_mha_scores']:
                    eval_output[eval_key] += [mha_score_sample.detach().cpu().numpy() for mha_score_sample in model_output[eval_key]]
                else:
                    eval_output[eval_key] += model_output[eval_key] if not hasattr(model_output[eval_key], 'to') else model_output[eval_key].detach().cpu().numpy().tolist()

            # -- saving embeddings for further analysis
            if is_test and args.save_embeddings:
                for sample_id, embedding in zip(batch['sample_id'], model_output['embeddings']):
                    save_embedding(embedding.detach().cpu().numpy(), args.output_dir, sample_id)

    eval_output['informed_metadata'] = eval_loader.dataset.informed_metadata
    eval_output['loss'] = np.array(eval_output['loss']).mean()
    eval_output['acc'] = np.array(eval_output['acc']).mean() * 100.0

    return eval_output

def pipeline(args, config, return_dicts=False):

    # -- setting seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # -- task and feature filtering
    print(config)
    print(f'Using the following features: {config.features}')
    same_config = task_and_feature_filtering(config, args.filter_tasks, args.exclude_features)

    diff_eval = False
    if 'none' not in args.filter_evaluation_tasks:
        diff_eval = True
        evaluation_config = task_and_feature_filtering(config, args.filter_evaluation_tasks, args.exclude_features)

    # -- building model architecture
    global model, optimizer, scheduler, train_loader, val_loader
    model = build_model(config)
    print(model)
    print(f'Model Parameters: {sum([param.nelement() for param in model.parameters()])}')

    # -- loading model checkpoint
    if args.load_checkpoint:
        print(f'Loading checkpoint from {args.load_checkpoint} ...')
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

    # -- creating dataloaders
    train_loader = get_dataloader(same_config, args.training_dataset, is_training=True)

    val_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.validation_dataset, is_training=False)
    val_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    test_loader = get_dataloader(evaluation_config if diff_eval else same_config, args.test_dataset, is_training=False)
    test_loader.dataset.feature_norm_stats = train_loader.dataset.feature_norm_stats

    # -- training process
    if args.mode in ['training', 'both']:

        # -- defining the optimizer and its scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(config, model, train_loader)

        for epoch in range(1, config.training_settings['epochs']+1):
            train_stats = train(config)
            val_output = evaluate(config, val_loader)

            # -- saving model checkpoint
            save_checkpoint(model, args.output_dir, f'epoch_{str(epoch).zfill(3)}')
            print(f"Epoch {epoch}: TRAIN LOSS={round(train_stats['loss'],2)} TRAIN ACC={round(train_stats['acc'],4)}% || VAL LOSS={round(val_output['loss'],4)} | VAL ACC={round(val_output['acc'],2)}%")

    if args.mode in ['evaluation', 'both']:

        val_output = evaluate(config, val_loader)
        test_output = evaluate(config, test_loader, is_test=True)

        if args.save_output:
            save_model_output(val_output, args.output_dir, 'validation', save_attention_scores=args.save_attention_scores)
            save_model_output(test_output, args.output_dir, 'test', save_attention_scores=args.save_attention_scores)

        # -- displaying final report
        val_report = classification_report(
            val_output['labels'],
            val_output['preds'],
            target_names=config.class_names,
            output_dict=return_dicts,
        )

        test_report = classification_report(
            test_output['labels'],
            test_output['preds'],
            target_names=config.class_names,
            output_dict=return_dicts,
        )

    return val_report, test_report

if __name__ == "__main__":

    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Training and/or evaluation of models.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', required=True, type=str, help='Configuration file to build, train, and evaluate the model')

    parser.add_argument('--training-dataset', default='./splits/neurovoz/fold_0/train.csv', type=str, help='CSV file representing the training dataset')
    parser.add_argument('--validation-dataset', required=True, type=str, help='CSV file representing the validation dataset')
    parser.add_argument('--test-dataset', required=True, type=str, help='CSV file representing the test dataset')

    parser.add_argument('--mode', default='both', type=str, help='Choose between: "training", "evaluation", or "both"')

    parser.add_argument("--exclude-features", nargs='+', default=['none'], help="Choose the features you don't want to use for ablation studies.")
    parser.add_argument('--filter-tasks', nargs='+', default=['ALL'], type=str, help='Choose the task to train')
    parser.add_argument('--filter-evaluation-tasks', nargs='+', default=['none'], type=str, help='Choose the task to evaluate on, otherwise we evaluate on the --filter-tasks')

    parser.add_argument("--yaml-overrides", metavar="CONF:[KEY]:VALUE", nargs='*', help="Set a number of conf-key-value pairs for modifying the yaml config file on the fly.")

    parser.add_argument('--load-checkpoint', default='', type=str, help='Choose between: "training", "evaluation", or "both"')
    parser.add_argument('--save-output', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--save-attention-scores', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--save-embeddings', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--output-dir', required=True, type=str, help='Path where to save model checkpoints and predictions')

    args = parser.parse_args()

    # -- loading configuration file
    config_file = Path(args.config)
    with config_file.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config = override_yaml(config, args.yaml_overrides)
    config = argparse.Namespace(**config)

    # -- pipeline process
    val_report, test_report = pipeline(args, config)

    print('\n--- VALIDATION ---')
    print(val_report)

    print('\n--- TEST ---')
    print(test_report)
