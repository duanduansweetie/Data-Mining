"""
图像异常检测 - 使用统一多模态框架
Image Anomaly Detection using Unified Multi-Modal Framework

本脚本使用统一框架进行图像异常检测，支持多种算法：
- mahalanobis: 马氏距离（默认，适合深度特征）
- isolation_forest: 孤立森林
- one_class_svm: One-Class SVM
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# 导入统一框架
from multimodal_anomaly_detection import create_image_detector

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def gather_files(data_root, class_name, split):
    """收集图像文件和标签"""
    base = os.path.join(data_root, class_name, split)
    files = []
    if not os.path.isdir(base):
        return files

    name_label_pairs = [('normal', 0), ('good', 0), ('anomaly', 1), ('bad', 1)]
    seen = set()

    for name, label in name_label_pairs:
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                p = os.path.join(d, fname)
                if p in seen:
                    continue
                seen.add(p)
                files.append((p, label))

    # 如果没有子目录，直接从base读取
    if len(files) == 0:
        direct_imgs = []
        for fname in os.listdir(base):
            p = os.path.join(base, fname)
            if os.path.isfile(p) and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                direct_imgs.append(p)

        if direct_imgs:
            for p in direct_imgs:
                fname = os.path.basename(p).lower()
                if any(k in fname for k in ('good', 'normal')):
                    label = 0
                elif any(k in fname for k in ('bad', 'anomaly', 'defect')):
                    label = 1
                else:
                    label = -1
                files.append((p, label))

    return files


def evaluate_class(data_root, class_name, detector, args):
    """评估单个类别的异常检测"""
    print(f"\n{'='*60}")
    print(f"处理类别: {class_name}")
    print(f"{'='*60}")

    # 收集训练数据
    train_files = gather_files(data_root, class_name, 'train')
    train_norm_all = [(f, l) for f, l in train_files if l == 0]

    if len(train_norm_all) == 0:
        print(f"错误: 没有找到训练集 normal/good 样本")
        return None, None

    print(f"训练样本数: {len(train_norm_all)}")

    # 收集测试数据
    test_files = gather_files(data_root, class_name, 'test')

    # 如果没有测试集，从训练集划分
    if len(test_files) == 0:
        print("未找到测试集，从训练集划分20%作为测试集...")
        rng = np.random.default_rng(0)
        n_good = len(train_norm_all)
        val_ratio = 0.2
        val_size = max(1, int(n_good * val_ratio))
        perm = rng.permutation(n_good)
        val_idxs = perm[:val_size].tolist()
        train_idxs = perm[val_size:].tolist()

        train_norm = [train_norm_all[i][0] for i in train_idxs]
        test_files = [train_norm_all[i] for i in val_idxs]  # 保留(path, label)格式
        print(f"训练集: {len(train_norm)}, 测试集: {len(test_files)}")
    else:
        train_norm = [f for f, l in train_norm_all]
        print(f"测试样本数: {len(test_files)}")

    # 正确提取路径和标签
    test_paths = [f for f, l in test_files]
    test_labels = np.array([l for f, l in test_files])

    # 训练模型（使用统一框架）
    print(f"\n使用 {args.method} 算法训练模型...")
    detector.fit_train(train_norm)

    # 计算异常分数
    print("计算异常分数...")
    scores = detector.detect(test_paths)

    # 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'{class_name}_scores.csv')
    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('path,label,score\n')
        for p, l, s in zip(test_paths, test_labels, scores):
            fh.write(f'"{p}",{int(l)},{s:.6f}\n')

    print(f"分数已保存到: {csv_path}")

    # 如果有标签，计算评估指标
    labeled_idx = [i for i, l in enumerate(test_labels) if l != -1]
    auroc = None

    if len(labeled_idx) > 0:
        lbls = test_labels[labeled_idx]
        scs = scores[labeled_idx]

        if len(np.unique(lbls)) >= 2:
            from sklearn.metrics import roc_auc_score
            try:
                auroc = roc_auc_score(lbls, scs)
                print(f"\n结果:")
                print(f"  AUROC: {auroc:.4f}")

                # 计算其他指标
                threshold = np.percentile(scs[lbls == 0], 95)
                predictions = (scs > threshold).astype(int)

                from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
                precision, recall, f1, _ = precision_recall_fscore_support(
                    lbls, predictions, average='binary', zero_division=0
                )
                cm = confusion_matrix(lbls, predictions)

                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Threshold: {threshold:.4f}")
                print(f"  混淆矩阵:")
                print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
                print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
            except Exception as e:
                print(f"计算指标时出错: {e}")
        else:
            print("标注子集不包含两类，无法计算评估指标")

    # 可视化top-k异常图像
    try:
        topk = min(8, len(scores))
        idxs = np.argsort(scores)[-topk:][::-1]

        fig, axes = plt.subplots(1, topk, figsize=(3 * topk, 3))
        if topk == 1:
            axes = [axes]

        for ax, i in zip(axes, idxs):
            img = Image.open(test_paths[i]).convert('RGB')
            ax.imshow(img)
            lab = test_labels[i]
            lab_str = str(int(lab)) if lab != -1 else 'unknown'
            ax.set_title(f'label={lab_str}\nscore={scores[i]:.2f}', fontsize=10)
            ax.axis('off')

        plt.suptitle(f'{class_name} - Top {topk} 异常样本', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'{class_name}_top_anom.png'))
        plt.close()
        print(f"可视化已保存到: {args.out_dir}/{class_name}_top_anom.png")
    except Exception as e:
        print(f"可视化时出错: {e}")

    return auroc, (test_labels, scores, test_paths)


def main():
    parser = argparse.ArgumentParser(description='图像异常检测 - 统一框架版本')
    parser.add_argument('--data_root', required=True, help='数据集根目录')
    parser.add_argument('--out_dir', default='./dataset', help='输出目录')
    parser.add_argument('--method', default='mahalanobis',
                       choices=['mahalanobis', 'isolation_forest', 'one_class_svm'],
                       help='异常检测算法')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备 (cuda/cpu)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("图像异常检测 - 统一多模态框架")
    print(f"{'='*60}")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.out_dir}")
    print(f"检测算法: {args.method}")
    print(f"计算设备: {args.device}")
    print(f"{'='*60}\n")

    # 检查数据目录
    if not os.path.exists(args.data_root):
        print(f"错误: 数据目录不存在: {args.data_root}")
        return

    # 自动处理嵌套目录结构
    classes = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    classes = [d for d in classes if d != '__MACOSX']

    if len(classes) == 1 and classes[0] == os.path.basename(os.path.normpath(args.data_root)):
        nested_root = os.path.join(args.data_root, classes[0])
        print(f"检测到嵌套目录结构，自动使用: {nested_root}")
        args.data_root = nested_root
        classes = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
        classes = [d for d in classes if d != '__MACOSX']

    classes = sorted(classes)

    if not classes:
        print("错误: 在 data_root 中未发现类目录")
        return

    print(f"发现类别: {classes}\n")

    # 创建检测器（使用统一框架）
    detector = create_image_detector(
        method=args.method,
        device=args.device
    )

    # 处理每个类别
    results = {}
    for class_name in classes:
        auroc, _ = evaluate_class(args.data_root, class_name, detector, args)
        if auroc is not None:
            results[class_name] = auroc

    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")

    if len(results) > 0:
        print(f"\n各类别 AUROC:")
        for class_name, auroc in results.items():
            print(f"  {class_name}: {auroc:.4f}")

        avg_auc = np.mean(list(results.values()))
        print(f"\n平均 AUROC: {avg_auc:.4f}")
    else:
        print("\n未计算出 AUROC（所有类测试集均无足够已标注样本）")
        print("输出目录中包含每类的分数 CSV，可用来后续分析")

    print(f"\n结果已保存到: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
