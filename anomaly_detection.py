import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt


class ImageFolderPairs(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path


def gather_files(data_root, class_name, split):
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
                elif any(k in fname for k in ('bad', 'anomaly', 'defect', 'anomal')):
                    label = 1
                else:
                    label = -1
                files.append((p, label))
    return files


def build_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_feature_extractor(device):
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet50(pretrained=True)
    modules = list(model.children())[:-1]
    feat_extractor = torch.nn.Sequential(*modules).to(device)
    feat_extractor.eval()
    return feat_extractor


def extract_features(dataloader, model, device):
    feats = []
    labels = []
    paths = []
    with torch.no_grad():
        for imgs, labs, pths in tqdm(dataloader, desc="extract"):
            imgs = imgs.to(device)
            out = model(imgs)
            out = out.view(out.size(0), -1).cpu().numpy()
            feats.append(out)
            labels.extend(labs.numpy().tolist())
            paths.extend(pths)
    if len(feats) == 0:
        return np.empty((0,)), np.array([], dtype=int), []
    feats = np.vstack(feats)
    return feats, np.array(labels), paths


def fit_gaussian(features, reg=1e-6):
    cov = EmpiricalCovariance(assume_centered=False).fit(features)
    cov_mat = cov.covariance_
    cov_mat += np.eye(cov_mat.shape[0]) * reg
    cov_inv = np.linalg.inv(cov_mat)
    mean = cov.location_
    return mean, cov_inv


def mahalanobis_score(x, mean, cov_inv):
    d = x - mean
    if d.ndim == 1:
        return float(d.dot(cov_inv).dot(d))
    return np.sum((d.dot(cov_inv)) * d, axis=1)


def evaluate_class(data_root, class_name, feat_extractor, device, args):
    transform = build_transform()
    base_train = os.path.join(data_root, class_name, 'train')
    if not os.path.isdir(base_train):
        print(f"缺失训练目录: {base_train}")
        raise SystemExit(1)

    train_files = gather_files(data_root, class_name, 'train')
    n_normal = sum(1 for _ in train_files if _[1] == 0)
    n_anom = sum(1 for _ in train_files if _[1] == 1)
    print(f"train 文件统计 - normal/good: {n_normal}, anomaly/bad: {n_anom}")

    train_norm_all = [f for f in train_files if f[1] == 0]
    train_bad_all = [f for f in train_files if f[1] == 1]

    if len(train_norm_all) == 0:
        contents = {}
        for k in ['good', 'normal', 'bad', 'anomaly']:
            d = os.path.join(base_train, k)
            contents[k] = os.listdir(d)[:5] if os.path.isdir(d) else []
        print("未找到训练集 normal/good 样本。train 子目录前几项示例：")
        for k, v in contents.items():
            print(f"  {k}: {len(v)} 文件 (前5): {v}")
        raise SystemExit(1)

    base_test = os.path.join(data_root, class_name, 'test')
    print(f"检查 test 目录: {base_test}")
    if os.path.isdir(base_test):
        print("test 目录下子项示例（最多 20）：", os.listdir(base_test)[:20])
    else:
        print("test 目录不存在。")

    test_files = gather_files(data_root, class_name, 'test')
    if len(test_files) == 0:
        print(f"未在 {base_test} 发现测试样本，自动从 train 划分测试集。")
        rng = np.random.default_rng(0)
        n_good = len(train_norm_all)
        val_ratio = 0.2
        val_size = max(1, int(n_good * val_ratio))
        perm = rng.permutation(n_good)
        val_idxs = perm[:val_size].tolist()
        train_idxs = perm[val_size:].tolist()
        train_norm = [train_norm_all[i] for i in train_idxs]
        val_good = [train_norm_all[i] for i in val_idxs]
        test_files = val_good + train_bad_all
        print(f"划分结果：用于训练 good={len(train_norm)}；用于测试 good={len(val_good)}, bad={len(train_bad_all)}")
    else:
        train_norm = train_norm_all
        print(f"test 文件数: {len(test_files)} (normal/good + anomaly/bad)")

    if len(train_norm) == 0:
        print("训练集（good）被划分为空，无法训练。请检查数据或调整划分比例。")
        raise SystemExit(1)

    labels_in_test = [lbl for _, lbl in test_files]
    if any(lbl == -1 for lbl in labels_in_test):
        print("检测到部分测试样本标签无法通过文件名推断，尝试自动查找标签映射或掩码...")
        class_dir = os.path.join(data_root, class_name)
        csv_candidates = ['test_labels.csv', 'labels.csv', 'ground_truth.csv', 'annotations.csv', 'gt.csv']
        label_map = {}
        found_csv = None
        for name in csv_candidates:
            p = os.path.join(class_dir, name)
            if os.path.isfile(p):
                found_csv = p
                break
        if found_csv is not None:
            print(f"找到标签文件: {found_csv}，尝试解析...")
            try:
                with open(found_csv, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            fname = parts[0].strip()
                            lab = parts[1].strip()
                            try:
                                lab = int(lab)
                            except Exception:
                                lab = 0 if lab.lower() in ('good', 'normal', '0') else 1
                            label_map[os.path.basename(fname)] = lab
                new_test = []
                for p, l in test_files:
                    if l != -1:
                        new_test.append((p, l))
                        continue
                    bn = os.path.basename(p)
                    if bn in label_map:
                        new_test.append((p, label_map[bn]))
                    else:
                        new_test.append((p, -1))
                test_files = new_test
            except Exception as e:
                print("解析标签文件失败:", e)

        labels_in_test = [lbl for _, lbl in test_files]
        if any(lbl == -1 for lbl in labels_in_test):
            gt_candidates = ['ground_truth', 'groundtruth', 'gt', 'masks', 'mask', 'ground-truth']
            gt_dir = None
            for g in gt_candidates:
                cand = os.path.join(class_dir, g)
                if os.path.isdir(cand):
                    gt_dir = cand
                    break
            if gt_dir is not None:
                print(f"找到掩码目录: {gt_dir}，将基于掩码存在与否标注异常样本...")
                mask_files = set(os.listdir(gt_dir))
                new_test = []
                for p, l in test_files:
                    if l != -1:
                        new_test.append((p, l))
                        continue
                    base_name = os.path.splitext(os.path.basename(p))[0]
                    matched = False
                    for mf in mask_files:
                        if base_name in mf:
                            matched = True
                            break
                    new_test.append((p, 1 if matched else 0))
                test_files = new_test

        labels_in_test = [lbl for _, lbl in test_files]
        if any(lbl == -1 for lbl in labels_in_test):
            print("自动推断后仍存在未标注的测试样本，示例（前10）：")
            print([p for p, l in test_files[:10] if l == -1])
            print("测试集中包含未标注样本；脚本将继续进行打分，随后保存包含标签(-1 表示未知)的 CSV 和 top-k 图像。")

    train_ds = ImageFolderPairs(train_norm, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    train_feats, _, _ = extract_features(train_loader, feat_extractor, device)
    if train_feats.size == 0 or train_feats.ndim != 2:
        print("提取到的训练特征为空或格式不对，无法拟合高斯分布。请检查训练集路径和图片格式。")
        raise SystemExit(1)

    mean, cov_inv = fit_gaussian(train_feats, reg=1e-6)

    test_ds = ImageFolderPairs(test_files, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_feats, test_labels, test_paths = extract_features(test_loader, feat_extractor, device)
    if test_feats.size == 0 or test_feats.ndim != 2:
        print("测试集特征为空，跳过该类别评估。")
        raise SystemExit(1)

    scores = mahalanobis_score(test_feats, mean, cov_inv)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, f'{class_name}_gauss.npz'), mean=mean, cov_inv=cov_inv)

    csv_path = os.path.join(args.out_dir, f'{class_name}_scores.csv')
    with open(csv_path, 'w', encoding='utf-8') as fh:
        fh.write('path,label,score\n')
        for p, l, s in zip(test_paths, test_labels, scores):
            fh.write(f'"{p}",{int(l) if l is not None else -1},{s:.6f}\n')

    labeled_idx = [i for i, l in enumerate(test_labels) if l != -1]
    auroc = None
    if len(labeled_idx) > 0:
        lbls = test_labels[labeled_idx]
        scs = scores[labeled_idx]
        if len(np.unique(lbls)) >= 2:
            try:
                auroc = roc_auc_score(lbls, scs)
                print(f"{class_name} AUROC (on labeled subset): {auroc:.4f}")
            except Exception as e:
                print("计算 AUROC 时出错:", e)
        else:
            print("标注子集不包含两类，无法计算 AUROC；将只保存分数 CSV 和 top-k 可视化。")
    else:
        print("测试集中没有任何已知标签；已保存分数 CSV 和 top-k 可视化，跳过 AUROC 计算。")

    topk = min(8, len(scores))
    idxs = np.argsort(scores)[-topk:][::-1]
    fig, axes = plt.subplots(1, topk, figsize=(3 * topk, 3))
    for ax, i in zip(axes, idxs):
        img = Image.open(test_paths[i]).convert('RGB')
        ax.imshow(img)
        lab = test_labels[i]
        lab_str = str(int(lab)) if lab != -1 else 'unknown'
        ax.set_title(f'label={lab_str} score={scores[i]:.2f}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'{class_name}_top_anom.png'))
    plt.close(fig)

    return auroc, (test_labels, scores, test_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='dataset root')
    parser.add_argument('--out_dir', default='results', help='output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print("使用 data_root:", args.data_root)
    print("输出目录 out_dir:", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat_extractor = get_feature_extractor(device)

    if not os.path.exists(args.data_root):
        print(f"数据目录不存在: {args.data_root}")
        raise SystemExit(1)

    classes = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    classes = sorted(classes)
    if not classes:
        print("在 data_root 中未发现类目录，请检查目录结构。")
        raise SystemExit(1)
    print("found classes:", classes)

    results = {}
    for c in classes:
        print(f"\nProcessing class {c} ...")
        auroc, _ = evaluate_class(args.data_root, c, feat_extractor, device, args)
        if auroc is None:
            print(f"{c}: 未计算 AUROC（无足够已标注样本）。已在输出目录保存每张测试图片的分数。")
            continue
        results[c] = auroc
        print(f"{c} AUROC: {auroc:.4f}")

    if len(results) > 0:
        avg = float(np.mean(list(results.values())))
        print("\nPer-class AUROC:", results)
        print("Average AUROC:", avg)
    else:
        print("\n未计算出任何 AUROC（所有类测试集均无足够已标注样本）。输出目录中包含每类的分数 CSV，可用来后续分析。")


if __name__ == '__main__':
    main()
