"""
表格数据异常检测 - 使用统一多模态框架
Tabular Data Anomaly Detection using Unified Multi-Modal Framework

甲状腺疾病异常检测任务
本脚本使用统一框架进行表格数据异常检测，支持多种算法：
- isolation_forest: 孤立森林（默认，适合表格数据）
- mahalanobis: 马氏距离
- one_class_svm: One-Class SVM
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

# 导入统一框架
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_anomaly_detection import create_tabular_detector

plt.style.use('seaborn-v0_8')


def load_data(train_csv, test_csv):
    """加载训练和测试数据"""
    print("加载数据...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"训练集规模: {train_df.shape}")
    print(f"测试集规模: {test_df.shape}")

    # 分离特征和标签
    train_data = train_df.values
    test_data = test_df.iloc[:, :-1].values
    test_labels = test_df.iloc[:, -1].values

    print(f"特征维度: {train_data.shape[1]}")
    print(f"训练集标签: 全部为正常（0）")
    print(f"测试集标签分布: 正常={np.sum(test_labels==0)}, 异常={np.sum(test_labels==1)}")

    return train_data, test_data, test_labels


def visualize_data(train_data, test_data, test_labels, out_dir):
    """数据探索性可视化"""
    print("\n生成数据探索性可视化...")

    os.makedirs(out_dir, exist_ok=True)

    # 特征分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        ax.hist(train_data[:, i], bins=50, alpha=0.6, label='Train (Normal)', color='blue')
        ax.hist(test_data[test_labels == 0, i], bins=50, alpha=0.4, label='Test (Normal)', color='green')
        ax.hist(test_data[test_labels == 1, i], bins=50, alpha=0.4, label='Test (Anomaly)', color='red')
        ax.set_title(f'Feature {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_distributions.png'))
    print(f"  特征分布图已保存: {out_dir}/feature_distributions.png")
    plt.close()

    # 相关性热图
    fig, ax = plt.subplots(figsize=(10, 8))
    all_data = np.vstack([train_data, test_data])
    corr = np.corrcoef(all_data.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
    print(f"  相关性热图已保存: {out_dir}/correlation_heatmap.png")
    plt.close()


def train_and_evaluate(train_data, test_data, test_labels, args):
    """训练和评估模型"""
    print(f"\n{'='*60}")
    print(f"使用 {args.method} 算法训练模型...")
    print(f"{'='*60}\n")

    # 创建检测器（使用统一框架）
    if args.method == 'isolation_forest':
        detector = create_tabular_detector(
            method='isolation_forest',
            contamination=args.contamination,
            n_estimators=args.n_estimators,
            random_state=42
        )
    elif args.method == 'mahalanobis':
        detector = create_tabular_detector(
            method='mahalanobis',
            reg=args.reg
        )
    elif args.method == 'one_class_svm':
        detector = create_tabular_detector(
            method='one_class_svm',
            nu=args.nu
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 训练模型
    print("训练中...")
    detector.fit_train(train_data)
    print("训练完成！\n")

    # 评估模型
    print("评估模型...")
    metrics = detector.evaluate(test_data, test_labels)

    # 输出结果
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"\n混淆矩阵:")
    print(f"              预测正常    预测异常")
    print(f"实际正常      {metrics['confusion_matrix'][0,0]:4d}       {metrics['confusion_matrix'][0,1]:4d}")
    print(f"实际异常      {metrics['confusion_matrix'][1,0]:4d}       {metrics['confusion_matrix'][1,1]:4d}")
    print(f"{'='*60}\n")

    return detector, metrics


def visualize_results(detector, test_data, test_labels, metrics, out_dir):
    """结果可视化"""
    print("生成结果可视化...")

    os.makedirs(out_dir, exist_ok=True)

    # 获取异常分数
    scores = detector.detect(test_data)

    # ROC曲线
    fpr, tpr, _ = roc_curve(test_labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {metrics["auc_roc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
    print(f"  ROC曲线已保存: {out_dir}/roc_curve.png")
    plt.close()

    # Precision-Recall曲线
    precision, recall, _ = precision_recall_curve(test_labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {metrics["auc_roc"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'))
    print(f"  PR曲线已保存: {out_dir}/pr_curve.png")
    plt.close()

    # 混淆矩阵热图
    cm = metrics['confusion_matrix']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    print(f"  混淆矩阵已保存: {out_dir}/confusion_matrix.png")
    plt.close()

    # 异常分数分布
    plt.figure(figsize=(10, 6))
    plt.hist(scores[test_labels == 0], bins=50, alpha=0.6, label='Normal', color='green')
    plt.hist(scores[test_labels == 1], bins=50, alpha=0.6, label='Anomaly', color='red')
    plt.axvline(metrics['threshold'], color='blue', linestyle='--', linewidth=2,
                label=f'Threshold = {metrics["threshold"]:.2f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'score_distribution.png'))
    print(f"  分数分布已保存: {out_dir}/score_distribution.png")
    plt.close()

    # 详细分类报告
    predictions = (scores > metrics['threshold']).astype(int)
    report = classification_report(test_labels, predictions,
                                   target_names=['Normal', 'Anomaly'],
                                   output_dict=True)

    print("\n详细分类报告:")
    print(f"              Precision  Recall  F1-Score  Support")
    print(f"Normal        {report['Normal']['precision']:.4f}    {report['Normal']['recall']:.4f}  {report['Normal']['f1-score']:.4f}     {report['Normal']['support']}")
    print(f"Anomaly       {report['Anomaly']['precision']:.4f}    {report['Anomaly']['recall']:.4f}  {report['Anomaly']['f1-score']:.4f}     {report['Anomaly']['support']}")
    print(f"\nAccuracy      {report['accuracy']:.4f}")
    print(f"Macro Avg     {report['macro avg']['precision']:.4f}    {report['macro avg']['recall']:.4f}  {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg  {report['weighted avg']['precision']:.4f}    {report['weighted avg']['recall']:.4f}  {report['weighted avg']['f1-score']:.4f}")

    # 保存预测结果
    results_df = pd.DataFrame({
        'true_label': test_labels,
        'predicted_label': predictions,
        'anomaly_score': scores
    })
    results_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    print(f"\n预测结果已保存: {out_dir}/predictions.csv")


def compare_methods(train_data, test_data, test_labels, out_dir):
    """比较不同方法"""
    print(f"\n{'='*60}")
    print("算法比较")
    print(f"{'='*60}\n")

    methods = [
        ('Isolation Forest', {
            'method': 'isolation_forest',
            'contamination': 0.01,
            'n_estimators': 200,
            'random_state': 42
        }),
        ('Mahalanobis', {
            'method': 'mahalanobis',
            'reg': 1e-6
        }),
        ('One-Class SVM', {
            'method': 'one_class_svm',
            'nu': 0.1
        })
    ]

    results = []

    for name, params in methods:
        print(f"\n训练 {name}...")
        detector = create_tabular_detector(**params)
        detector.fit_train(train_data)
        metrics = detector.evaluate(test_data, test_labels)

        results.append({
            'Method': name,
            'AUC-ROC': metrics['auc_roc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1']
        })

        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")

    # 保存比较结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, 'method_comparison.csv'), index=False)

    print(f"\n{'='*60}")
    print("算法比较总结:")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"{'='*60}")

    # 可视化比较
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics_names = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']

    for ax, metric_name in zip(axes, metrics_names):
        values = results_df[metric_name].values
        ax.bar(results_df['Method'], values, color=['blue', 'green', 'orange'])
        ax.set_title(metric_name)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'method_comparison.png'))
    print(f"\n算法比较图已保存: {out_dir}/method_comparison.png")
    plt.close()

    return results_df


def main():
    parser = argparse.ArgumentParser(description='表格数据异常检测 - 统一框架版本')
    parser.add_argument('--train_csv', default='./train-set.csv',
                       help='训练集CSV文件路径')
    parser.add_argument('--test_csv', default='./test-set.csv',
                       help='测试集CSV文件路径')
    parser.add_argument('--out_dir', default='./results_tabular',
                       help='输出目录')
    parser.add_argument('--method', default='isolation_forest',
                       choices=['isolation_forest', 'mahalanobis', 'one_class_svm'],
                       help='异常检测算法')

    # Isolation Forest 参数
    parser.add_argument('--contamination', type=float, default=0.01,
                       help='Isolation Forest污染率')
    parser.add_argument('--n_estimators', type=int, default=200,
                       help='Isolation Forest树的数量')

    # Mahalanobis 参数
    parser.add_argument('--reg', type=float, default=1e-6,
                       help='马氏距离正则化参数')

    # One-Class SVM 参数
    parser.add_argument('--nu', type=float, default=0.1,
                       help='One-Class SVM nu参数')

    # 其他选项
    parser.add_argument('--compare', action='store_true',
                       help='是否运行算法比较')
    parser.add_argument('--skip_viz', action='store_true',
                       help='跳过可视化')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("表格数据异常检测 - 统一多模态框架")
    print(f"{'='*60}")
    print(f"训练集: {args.train_csv}")
    print(f"测试集: {args.test_csv}")
    print(f"输出目录: {args.out_dir}")
    print(f"检测算法: {args.method}")
    print(f"{'='*60}\n")

    # 检查文件是否存在
    if not os.path.exists(args.train_csv):
        print(f"错误: 训练文件不存在: {args.train_csv}")
        return

    if not os.path.exists(args.test_csv):
        print(f"错误: 测试文件不存在: {args.test_csv}")
        return

    # 加载数据
    train_data, test_data, test_labels = load_data(args.train_csv, args.test_csv)

    # 数据探索性可视化
    if not args.skip_viz:
        visualize_data(train_data, test_data, test_labels, args.out_dir)

    # 训练和评估模型
    detector, metrics = train_and_evaluate(train_data, test_data, test_labels, args)

    # 结果可视化
    if not args.skip_viz:
        visualize_results(detector, test_data, test_labels, metrics, args.out_dir)

    # 算法比较
    if args.compare:
        compare_methods(train_data, test_data, test_labels, args.out_dir)

    print(f"\n{'='*60}")
    print("完成！")
    print(f"所有结果已保存到: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
