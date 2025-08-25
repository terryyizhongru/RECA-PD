import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def compute_average_report_across_runs(reports):
    overall_report = {
        'HC': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'PD': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'accuracy': [],
        'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    }

    # Spoiler: It's gonna be inefficient :)

    for report in reports:
        for key in report.keys():
            if key == 'accuracy':
                overall_report[key].append(report[key])
            else:
                for key2 in report[key].keys():
                    overall_report[key][key2].append(report[key][key2])

    for key in overall_report.keys():
            if key == 'accuracy':
                overall_report[key] = f'{round(np.array(overall_report[key]).mean(), 4)}±{round(np.array(overall_report[key]).std(), 4)}'
            else:
                for key2 in report[key].keys():
                    overall_report[key][key2] = f'{round(np.array(overall_report[key][key2]).mean(), 4)}±{round(np.array(overall_report[key][key2]).std(), 4)}'

    # -- just for a more clean output
    overall_report = pd.DataFrame.from_dict(overall_report).T
    overall_report.iloc[2,0] = ''
    overall_report.iloc[2,1] = ''
    overall_report.iloc[2,3] = overall_report.iloc[3,3]

    return overall_report

def get_reports(exps_dir):
    val_reports = []
    test_reports = []
    run_dirs = os.listdir(exps_dir)
    for run_dir in run_dirs:
        val_preds = []
        val_labels = []
        test_preds = []
        test_labels = []

        run_dir_path = os.path.join(exps_dir, run_dir)
        fold_dirs = os.listdir(run_dir_path)
        
        for fold_dir in fold_dirs:
            model_output_dir = os.path.join(run_dir_path, fold_dir, 'model_output')

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
        val_reports.append(
            classification_report(
                val_labels,
                val_preds,
                target_names=['HC', 'PD'],
                output_dict=True,
            )
        )

        test_reports.append(
            classification_report(
                test_labels,
                test_preds,
                target_names=['HC', 'PD'],
                output_dict=True,
            )
        )

    return val_reports, test_reports


def get_report_perfold(exps_dir):
    """
    获取每个run的每个fold的报告
    
    Args:
        exps_dir: 存储实验结果的目录路径
        
    Returns:
        val_fold_reports: 字典，键为run_dir，值为该run下每个fold的验证集报告列表
        test_fold_reports: 字典，键为run_dir，值为该run下每个fold的测试集报告列表
    """
    val_fold_reports = {}
    test_fold_reports = {}
    run_dirs = sorted(os.listdir(exps_dir))
    
    for run_dir in run_dirs:
        run_dir_path = os.path.join(exps_dir, run_dir)
        fold_dirs = sorted(os.listdir(run_dir_path))
        
        val_fold_reports[run_dir] = []
        test_fold_reports[run_dir] = []
        
        for fold_dir in fold_dirs:
            model_output_dir = os.path.join(run_dir_path, fold_dir, 'model_output')
            
            # -- validation set
            val_report_path = os.path.join(model_output_dir, 'validation_classification.pkl')
            if os.path.exists(val_report_path):
                with open(val_report_path, 'rb') as f:
                    val_model_output = pickle.load(f)
                
                # 提取概率值用于AUC计算
                val_probs = [preds[1] for preds in val_model_output['probs']]
                val_model_output['probs'] = val_probs  # 更新为单一概率值
                
                # 计算AUC
                val_auc = roc_auc_score(val_model_output['labels'], val_probs)
                
                # 计算单个fold的验证集分类报告
                val_fold_report = classification_report(
                    val_model_output['labels'],
                    val_model_output['preds'],
                    target_names=['HC', 'PD'],
                    output_dict=True,
                )
                
                val_fold_reports[run_dir].append({
                    'fold': fold_dir,
                    'report': val_fold_report,
                    'preds': val_model_output['preds'],
                    'labels': val_model_output['labels'],
                    'probs': val_model_output['probs'],  # 添加概率值
                    'auc': val_auc  # 添加AUC
                })
            
            # -- test set
            test_report_path = os.path.join(model_output_dir, 'test_classification.pkl')
            if os.path.exists(test_report_path):
                with open(test_report_path, 'rb') as f:
                    test_model_output = pickle.load(f)
                
                # 提取概率值用于AUC计算
                test_probs = [preds[1] for preds in test_model_output['probs']]
                test_model_output['probs'] = test_probs  # 更新为单一概率值
                
                # 计算AUC
                test_auc = roc_auc_score(test_model_output['labels'], test_probs)
                
                # 计算单个fold的测试集分类报告
                test_fold_report = classification_report(
                    test_model_output['labels'],
                    test_model_output['preds'],
                    target_names=['HC', 'PD'],
                    output_dict=True,
                )
                
                test_fold_reports[run_dir].append({
                    'fold': fold_dir,
                    'report': test_fold_report,
                    'preds': test_model_output['preds'],
                    'labels': test_model_output['labels'],
                    'probs': test_model_output['probs'],  # 添加概率值
                    'auc': test_auc  # 添加AUC
                })
    
    return val_fold_reports, test_fold_reports

def compute_fold_stats(fold_reports):
    """
    计算每个run中所有fold的性能指标的均值和标准差
    
    Args:
        fold_reports: 字典，键为run_dir，值为该run下每个fold的报告列表
        
    Returns:
        fold_stats: 字典，键为run_dir，值为包含accuracy、f1、precision、recall、sensitivity、specificity、auc的均值和标准差的字典
    """
    fold_stats = {}
    
    for run_dir, fold_results in fold_reports.items():
        accuracies = [fold_result['report']['accuracy'] for fold_result in fold_results]
        f1_scores = [fold_result['report']['PD']['f1-score'] for fold_result in fold_results]
        precisions = [fold_result['report']['PD']['precision'] for fold_result in fold_results]
        recalls = [fold_result['report']['PD']['recall'] for fold_result in fold_results]
        
        # 计算灵敏度(Sensitivity)和特异度(Specificity)
        sensitivities = [fold_result['report']['PD']['recall'] for fold_result in fold_results]  # 灵敏度就是PD的召回率
        specificities = [fold_result['report']['HC']['recall'] for fold_result in fold_results]  # 特异度是HC的召回率
        
        # 添加AUC指标
        aucs = [fold_result['auc'] for fold_result in fold_results]
        
        fold_stats[run_dir] = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            },
            'f1': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores)
            },
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions)
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls)
            },
            'sensitivity': {
                'mean': np.mean(sensitivities),
                'std': np.std(sensitivities)
            },
            'specificity': {
                'mean': np.mean(specificities),
                'std': np.std(specificities)
            },
            'auc': {
                'mean': np.mean(aucs),
                'std': np.std(aucs)
            }
        }
    
    return fold_stats

if __name__ == '__main__':
    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Performance analysis across runs and folds.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where the experiments are stored.')

    args = parser.parse_args()

    # 获取每个fold的报告
    val_fold_reports, test_fold_reports = get_report_perfold(args.exps_dir)
    
    # 计算每个run的统计数据
    # val_fold_stats = compute_fold_stats(val_fold_reports)
    test_fold_stats = compute_fold_stats(test_fold_reports)
    
    # # 打印每个run每个fold的结果
    # print("\nValidation Performance Per Fold:")
    # for run_dir, fold_results in val_fold_reports.items():
    #     print(f"\nRun: {run_dir}")
    #     for fold_result in fold_results:
    #         fold = fold_result['fold']
    #         report = fold_result['report']
    #         print(f"  Fold: {fold}, Accuracy: {report['accuracy']:.4f}, F1 (PD): {report['PD']['f1-score']:.4f}")
        
    #     # 打印该run的所有fold的均值和标准差
    #     # 打印该run的所有fold的均值和标准差
    #         stats = val_fold_stats[run_dir]
    #         print(f"  Mean Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
    #         print(f"  Mean F1 (PD): {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
    #         print(f"  Mean Precision (PD): {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
    #         print(f"  Mean Recall (PD): {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
    #         print(f"  Mean Sensitivity: {stats['sensitivity']['mean']:.4f} ± {stats['sensitivity']['std']:.4f}")
    #         print(f"  Mean Specificity: {stats['specificity']['mean']:.4f} ± {stats['specificity']['std']:.4f}")
    
    # 打印每个run每个fold的结果
    # print("\nTest Performance Per Fold:")
    for run_dir, fold_results in test_fold_reports.items():
        # print(f"\nRun: {run_dir}")
        for fold_result in fold_results:
            fold = fold_result['fold']
            report = fold_result['report']
            # print(f"  Fold: {fold}, Accuracy: {report['accuracy']:.4f}, F1 (PD): {report['PD']['f1-score']:.4f}")
        
        # 打印该run的所有fold的均值和标准差
        stats = test_fold_stats[run_dir]
        # 打印该run的所有fold的均值和标准差
        # print(f"  Mean Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        # print(f"  Mean F1 (PD): {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
        # print(f"  Mean Precision (PD): {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
        # print(f"  Mean Recall (PD): {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
        # print(f"  Mean Sensitivity: {stats['sensitivity']['mean']:.4f} ± {stats['sensitivity']['std']:.4f}")
        # print(f"  Mean Specificity: {stats['specificity']['mean']:.4f} ± {stats['specificity']['std']:.4f}")
    
    # 计算所有runs的平均性能指标和标准差
    # print("\n=== Average Performance Across All Test Runs ===")
    all_run_accuracies = [stats['accuracy']['mean'] for stats in test_fold_stats.values()]
    all_run_f1s = [stats['f1']['mean'] for stats in test_fold_stats.values()]
    all_run_precisions = [stats['precision']['mean'] for stats in test_fold_stats.values()]
    all_run_recalls = [stats['recall']['mean'] for stats in test_fold_stats.values()]
    all_run_sensitivities = [stats['sensitivity']['mean'] for stats in test_fold_stats.values()]
    all_run_specificities = [stats['specificity']['mean'] for stats in test_fold_stats.values()]

    # 计算所有runs的标准差的平均值
    all_run_accuracies_stds = [stats['accuracy']['std'] for stats in test_fold_stats.values()]
    all_run_f1s_stds = [stats['f1']['std'] for stats in test_fold_stats.values()]
    all_run_precisions_stds = [stats['precision']['std'] for stats in test_fold_stats.values()]
    all_run_recalls_stds = [stats['recall']['std'] for stats in test_fold_stats.values()]
    all_run_sensitivities_stds = [stats['sensitivity']['std'] for stats in test_fold_stats.values()]
    all_run_specificities_stds = [stats['specificity']['std'] for stats in test_fold_stats.values()]
    # 汇总统计部分也需要加入AUC
    all_run_aucs = [stats['auc']['mean'] for stats in test_fold_stats.values()]
    all_run_aucs_stds = [stats['auc']['std'] for stats in test_fold_stats.values()]

    # 打印所有runs的平均性能
    # print(f"Mean Accuracy across runs: {np.mean(all_run_accuracies):.4f} ± {np.std(all_run_accuracies):.4f}")
    # print(f"Mean F1 (PD) across runs: {np.mean(all_run_f1s):.4f} ± {np.std(all_run_f1s):.4f}")
    # print(f"Mean Precision (PD) across runs: {np.mean(all_run_precisions):.4f} ± {np.std(all_run_precisions):.4f}")
    # print(f"Mean Recall (PD) across runs: {np.mean(all_run_recalls):.4f} ± {np.std(all_run_recalls):.4f}")
    # print(f"Mean Sensitivity across runs: {np.mean(all_run_sensitivities):.4f} ± {np.std(all_run_sensitivities):.4f}")
    # print(f"Mean Specificity across runs: {np.mean(all_run_specificities):.4f} ± {np.std(all_run_specificities):.4f}")
    # print(f"Mean AUC across runs: {np.mean(all_run_aucs):.4f} ± {np.std(all_run_aucs):.4f}")


    print(args.exps_dir.strip("/").split("/")[-1], ": F1:", f"{np.mean(all_run_f1s):.4f} ± {np.mean(all_run_f1s_stds):.4f}")

