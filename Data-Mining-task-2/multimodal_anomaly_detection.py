"""
多模态统一异常检测框架
Unified Multi-Modal Anomaly Detection Framework

该框架支持同时处理图像和表格数据的异常检测任务。
核心思想：将特征提取和异常检测解耦，提供统一的接口。

支持的数据模态：
1. 图像数据（Image） - 使用深度学习模型提取特征
2. 表格数据（Tabular） - 直接使用原始特征或预处理

支持的异常检测算法：
1. 马氏距离（Mahalanobis Distance）
2. 孤立森林（Isolation Forest）
3. One-Class SVM
"""

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 特征提取器
# ============================================

class FeatureExtractor(ABC):
    """特征提取器抽象基类"""

    @abstractmethod
    def fit(self, data: Union[np.ndarray, List[str]]) -> 'FeatureExtractor':
        """学习数据特征（如标准化参数）"""
        pass

    @abstractmethod
    def transform(self, data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """提取特征"""
        pass

    def fit_transform(self, data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """拟合并转换数据"""
        self.fit(data)
        return self.transform(data)


class ImageFeatureExtractor(FeatureExtractor):
    """图像特征提取器 - 使用预训练CNN模型"""

    def __init__(self, model_name='resnet50', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()
        # 不再保存self.transform，每次使用时创建新的
        self.scaler = StandardScaler()

    def _load_model(self):
        """加载预训练模型"""
        if self.model_name == 'resnet50':
            try:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except:
                model = models.resnet50(pretrained=True)

        # 移除最后的分类层，只保留特征提取部分
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules).to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        """构建图像预处理变换"""
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_raw_features(self, image_paths: List[str]) -> np.ndarray:
        """从图像路径提取原始特征"""
        if not image_paths:
            return np.empty((0, 2048))  # ResNet50 feature dimension

        class ImageDataset(Dataset):
            def __init__(self, paths, base_transform):
                self.paths = paths
                self.base_transform = base_transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                path = self.paths[idx]
                try:
                    img = Image.open(path).convert('RGB')
                    # 应用变换
                    if self.base_transform:
                        img = self.base_transform(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # 返回一个默认张量
                    img = torch.zeros(3, 224, 224)
                return img

        try:
            # 创建transform
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            dataset = ImageDataset(image_paths, transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            features = []
            with torch.no_grad():
                for batch_imgs in loader:
                    # 确保是张量
                    if not isinstance(batch_imgs, torch.Tensor):
                        print(f"Warning: Unexpected batch type {type(batch_imgs)}")
                        continue

                    batch_imgs = batch_imgs.to(self.device)
                    feats = self.model(batch_imgs)
                    feats = feats.view(feats.size(0), -1).cpu().numpy()
                    features.append(feats)

            if features:
                return np.vstack(features)
            else:
                return np.empty((0, 2048))
        except Exception as e:
            print(f"Error in _extract_raw_features: {e}")
            import traceback
            traceback.print_exc()
            return np.empty((0, 2048))

    def fit(self, image_paths: List[str]) -> 'ImageFeatureExtractor':
        """拟合标准化器"""
        features = self._extract_raw_features(image_paths)
        if features.size > 0:
            self.scaler.fit(features)
        return self

    def transform(self, image_paths: List[str]) -> np.ndarray:
        """提取并标准化特征"""
        features = self._extract_raw_features(image_paths)
        if features.size > 0:
            features = self.scaler.transform(features)
        return features


class TabularFeatureExtractor(FeatureExtractor):
    """表格数据特征提取器 - 标准化处理"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data: np.ndarray) -> 'TabularFeatureExtractor':
        """拟合标准化器"""
        self.scaler.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """标准化特征"""
        return self.scaler.transform(data)


# ============================================
# 异常检测器
# ============================================

class AnomalyDetector(ABC):
    """异常检测器抽象基类"""

    @abstractmethod
    def fit(self, features: np.ndarray) -> 'AnomalyDetector':
        """在正常数据上训练"""
        pass

    @abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray:
        """计算异常分数（分数越高越异常）"""
        pass

    def predict(self, features: np.ndarray, threshold: float = None) -> np.ndarray:
        """预测是否为异常（返回0/1）"""
        scores = self.score(features)
        if threshold is None:
            # 使用训练集的某个分位数作为阈值
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)


class MahalanobisDetector(AnomalyDetector):
    """基于马氏距离的异常检测器"""

    def __init__(self, reg=1e-6):
        self.reg = reg
        self.mean = None
        self.cov_inv = None

    def fit(self, features: np.ndarray) -> 'MahalanobisDetector':
        """拟合高斯分布"""
        cov = EmpiricalCovariance(assume_centered=False).fit(features)
        cov_mat = cov.covariance_.copy()
        cov_mat += np.eye(cov_mat.shape[0]) * self.reg
        self.cov_inv = np.linalg.inv(cov_mat)
        self.mean = cov.location_
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """计算马氏距离"""
        d = features - self.mean
        if d.ndim == 1:
            d = d.reshape(1, -1)
        return np.sum((d.dot(self.cov_inv)) * d, axis=1)


class IsolationForestDetector(AnomalyDetector):
    """基于孤立森林的异常检测器"""

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def fit(self, features: np.ndarray) -> 'IsolationForestDetector':
        """训练孤立森林"""
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """计算异常分数（转换为正值，越大越异常）"""
        # IsolationForest的score_samples返回负值，绝对值越大越异常
        scores = -self.model.score_samples(features)
        return scores


class OneClassSVMDetector(AnomalyDetector):
    """基于One-Class SVM的异常检测器"""

    def __init__(self, nu=0.1, kernel='rbf'):
        self.nu = nu
        self.kernel = kernel
        self.model = None

    def fit(self, features: np.ndarray) -> 'OneClassSVMDetector':
        """训练One-Class SVM"""
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel)
        self.model.fit(features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """计算异常分数（转换为正值，越大越异常）"""
        # decision_function返回有符号距离，负值表示异常
        scores = -self.model.decision_function(features)
        return scores


# ============================================
# 统一异常检测框架
# ============================================

class UnifiedAnomalyDetector:
    """
    统一多模态异常检测框架

    用法：
    1. 图像数据：
       detector = UnifiedAnomalyDetector(modality='image')
       detector.fit_train(normal_image_paths)
       scores = detector.detect(test_image_paths)

    2. 表格数据：
       detector = UnifiedAnomalyDetector(modality='tabular')
       detector.fit_train(normal_tabular_data)
       scores = detector.detect(test_tabular_data)
    """

    def __init__(self,
                 modality: str = 'tabular',
                 feature_extractor: FeatureExtractor = None,
                 anomaly_detector: AnomalyDetector = None,
                 device='cpu'):

        self.modality = modality.lower()
        self.device = device

        # 初始化特征提取器
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            if self.modality == 'image':
                self.feature_extractor = ImageFeatureExtractor(device=device)
            elif self.modality == 'tabular':
                self.feature_extractor = TabularFeatureExtractor()
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        # 初始化异常检测器
        if anomaly_detector is not None:
            self.anomaly_detector = anomaly_detector
        else:
            # 默认使用马氏距离
            self.anomaly_detector = MahalanobisDetector()

    def fit_train(self, train_data: Union[np.ndarray, List[str]], labels=None) -> 'UnifiedAnomalyDetector':
        """
        在训练数据上拟合模型

        参数：
        - train_data: 训练数据（图像路径列表或表格数据数组）
        - labels: 可选标签（用于有监督设置，当前为半监督/无监督）
        """
        print(f"[UnifiedAnomalyDetector] 拟合 {self.modality} 模态的特征提取器...")

        # 验证输入数据
        if isinstance(train_data, list):
            print(f"[DEBUG] train_data 是列表，长度: {len(train_data)}")
            if len(train_data) > 0:
                print(f"[DEBUG] 第一个元素类型: {type(train_data[0])}")
                if len(train_data) > 0 and isinstance(train_data[0], str):
                    print(f"[DEBUG] 第一个路径: {train_data[0]}")
        elif isinstance(train_data, np.ndarray):
            print(f"[DEBUG] train_data 是numpy数组，形状: {train_data.shape}")

        # 1. 拟合特征提取器
        try:
            self.feature_extractor.fit(train_data)
        except Exception as e:
            print(f"[ERROR] fit() 失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[UnifiedAnomalyDetector] 提取训练特征...")
        # 2. 提取训练特征
        try:
            train_features = self.feature_extractor.transform(train_data)
        except Exception as e:
            print(f"[ERROR] transform() 失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[UnifiedAnomalyDetector] 训练特征形状: {train_features.shape}")

        print(f"[UnifiedAnomalyDetector] 训练异常检测器...")
        # 3. 训练异常检测器
        try:
            self.anomaly_detector.fit(train_features)
        except Exception as e:
            print(f"[ERROR] anomaly_detector.fit() 失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"[UnifiedAnomalyDetector] 训练完成！")
        return self

    def detect(self, test_data: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        检测测试数据中的异常

        返回：
        - scores: 异常分数数组（越大越异常）
        """
        # 1. 提取测试特征
        test_features = self.feature_extractor.transform(test_data)

        # 2. 计算异常分数
        scores = self.anomaly_detector.score(test_features)

        return scores

    def predict(self, test_data: Union[np.ndarray, List[str]], threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测测试数据中的异常

        返回：
        - predictions: 0/1数组（1表示异常）
        - scores: 异常分数数组
        """
        scores = self.detect(test_data)
        predictions = self.anomaly_detector.predict(test_features=None, threshold=threshold)

        # 重新计算以获取正确的预测
        test_features = self.feature_extractor.transform(test_data)
        test_scores = self.anomaly_detector.score(test_features)

        if threshold is None:
            # 使用95分位数作为默认阈值
            threshold = np.percentile(test_scores, 95)

        predictions = (test_scores > threshold).astype(int)

        return predictions, test_scores

    def evaluate(self, test_data: Union[np.ndarray, List[str]],
                 true_labels: np.ndarray) -> dict:
        """
        评估模型性能

        参数：
        - test_data: 测试数据
        - true_labels: 真实标签（0=正常，1=异常）

        返回：
        - metrics: 包含各种评估指标的字典
        """
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

        # 获取异常分数
        scores = self.detect(test_data)

        # 计算AUC-ROC
        auc_roc = roc_auc_score(true_labels, scores)

        # 使用最佳阈值进行预测
        threshold = np.percentile(scores[true_labels == 0], 95)
        predictions = (scores > threshold).astype(int)

        # 计算其他指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )

        cm = confusion_matrix(true_labels, predictions)

        metrics = {
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'threshold': threshold
        }

        return metrics


# ============================================
# 工厂函数 - 便捷创建不同配置的检测器
# ============================================

def create_image_detector(method='mahalanobis', device='cpu', **kwargs):
    """创建图像异常检测器"""
    # 选择异常检测方法
    if method == 'mahalanobis':
        detector = MahalanobisDetector(**kwargs)
    elif method == 'isolation_forest':
        detector = IsolationForestDetector(**kwargs)
    elif method == 'one_class_svm':
        detector = OneClassSVMDetector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return UnifiedAnomalyDetector(
        modality='image',
        anomaly_detector=detector,
        device=device
    )


def create_tabular_detector(method='isolation_forest', **kwargs):
    """创建表格数据异常检测器"""
    # 选择异常检测方法
    if method == 'mahalanobis':
        detector = MahalanobisDetector(**kwargs)
    elif method == 'isolation_forest':
        detector = IsolationForestDetector(**kwargs)
    elif method == 'one_class_svm':
        detector = OneClassSVMDetector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return UnifiedAnomalyDetector(
        modality='tabular',
        anomaly_detector=detector
    )


# ============================================
# 示例使用
# ============================================

def example_image_detection():
    """图像异常检测示例"""
    print("\n=== 图像异常检测示例 ===\n")

    # 假设有图像路径
    train_images = ['path/to/normal1.jpg', 'path/to/normal2.jpg', ...]
    test_images = ['path/to/test1.jpg', 'path/to/test2.jpg', ...]

    # 创建检测器
    detector = create_image_detector(method='mahalanobis', device='cpu')

    # 训练
    detector.fit_train(train_images)

    # 检测
    scores = detector.detect(test_images)

    print(f"异常分数: {scores}")


def example_tabular_detection():
    """表格数据异常检测示例"""
    print("\n=== 表格数据异常检测示例 ===\n")

    # 加载数据
    train_df = pd.read_csv('train-set.csv')
    test_df = pd.read_csv('test-set.csv')

    # 假设前几列是特征
    train_data = train_df.iloc[:, :-1].values
    test_data = test_df.iloc[:, :-1].values
    true_labels = test_df.iloc[:, -1].values

    # 创建检测器
    detector = create_tabular_detector(
        method='isolation_forest',
        contamination=0.01,
        n_estimators=200
    )

    # 训练
    detector.fit_train(train_data)

    # 评估
    metrics = detector.evaluate(test_data, true_labels)

    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"混淆矩阵:\n{metrics['confusion_matrix']}")


if __name__ == '__main__':
    print("多模态统一异常检测框架")
    print("=" * 50)
    print("\n该框架支持：")
    print("1. 图像数据异常检测（使用深度特征）")
    print("2. 表格数据异常检测（使用原始特征）")
    print("3. 多种异常检测算法（马氏距离/孤立森林/One-Class SVM）")
    print("\n核心思想：将特征提取和异常检测解耦，实现统一接口。")
