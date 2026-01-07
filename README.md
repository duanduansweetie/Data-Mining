
## 项目概览

### Task 1: 基于深度特征的图像聚类分析
- **任务类型**: 无监督学习 - 图像聚类
- **核心算法**: ResNet50深度特征提取 + K-means聚类
- **数据集**: 工业产品图像数据集（6个类别，600张图片）
- **评估指标**: ARI=1.0, NMI=1.0（完美聚类）

### Task 3: 基于LSTM的温度时间序列预测
- **任务类型**: 有监督学习 - 时间序列回归预测
- **核心算法**: 双层LSTM神经网络
- **数据集**: 德国气象站气象数据（26,200条记录，21个特征）
- **评估指标**: R²>0.95, RMSE≈0.5°C

## 项目结构

```
.
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包列表
├── Task1/                              # Task1: 图像聚类分析
│   ├── task1.ipynb                     # Jupyter Notebook主文件
│   ├── cluster_labels.json             # 真实标签文件
│   └── dataset/                        # 图像数据集目录
│       ├── 001.png
│       ├── 002.png
│       └── ...
├── Task3/                              # Task3: 温度时间序列预测
│   ├── task3.ipynb                     # Jupyter Notebook主文件
│   ├── weather.csv                     # 气象数据集
│   ├── weather_lstm_model.h5           # 训练好的LSTM模型
│   ├── temperature_predictor.h5        # 温度预测器模型
│   └── temperature_prediction_results.png  # 预测结果可视化图
├── Cluster/                            # 聚类相关资源
├── thyroid/                            # 甲状腺数据集
└── weather.csv                         # 气象数据副本
```

## 环境要求

- Python 3.8+
- CUDA（可选，用于GPU加速）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Task 1 - 图像聚类分析

```bash
cd Task1
jupyter notebook task1.ipynb
```

**实验内容**:
- 使用ResNet50预训练模型提取图像深度特征
- 应用K-means算法对600张工业产品图像进行聚类
- 使用t-SNE降维可视化聚类结果
- 计算ARI和NMI评估聚类性能

**主要结果**:
- 特征维度: 2048维（ResNet50全局平均池化输出）
- 聚类数量: 6个簇（对应6个产品类别）
- ARI: 1.0000（完美聚类）
- NMI: 1.0000（完美聚类）

### 3. 运行Task 3 - 温度时间序列预测

```bash
cd Task3
jupyter notebook task3.ipynb
```

**实验内容**:
- 分析德国气象站气象时间序列数据
- 构建双层LSTM深度学习模型
- 训练模型并预测未来温度变化
- 多维度可视化预测结果和误差分析

**主要结果**:
- 模型架构: 双层LSTM + Dropout + Dense层
- 序列长度: 12个时间步（2小时）
- 训练集: 20,950个序列
- 测试集: 5,238个序列
- RMSE: ~0.5°C
- R²: >0.95

## 技术栈

### Task 1 技术栈
- **深度学习框架**: PyTorch, TorchVision
- **预训练模型**: ResNet50（ImageNet预训练权重）
- **聚类算法**: K-means (scikit-learn)
- **降维可视化**: t-SNE
- **数据处理**: NumPy, Pillow, OpenCV
- **可视化**: Matplotlib, Seaborn

### Task 3 技术栈
- **深度学习框架**: TensorFlow, Keras
- **神经网络架构**: 双层LSTM
- **数据处理**: Pandas, NumPy
- **数据预处理**: StandardScaler (scikit-learn)
- **评估指标**: MSE, MAE, R²
- **可视化**: Matplotlib
- **优化器**: Adam
- **正则化**: Dropout, Early Stopping

## 算法原理

### ResNet50特征提取
ResNet（Residual Network）通过残差连接解决深层网络梯度消失问题：
```
y = F(x, {W_i}) + x
```
使用预训练模型实现迁移学习，提取2048维深度特征。

### K-means聚类
基于划分的聚类算法，最小化簇内平方误差和（SSE）：
```
SSE = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

### LSTM时间序列预测
LSTM通过门控机制捕捉长期依赖关系：
- **遗忘门**: 控制丢弃信息
- **输入门**: 控制新信息写入
- **输出门**: 控制输出信息

## 数据集说明

### Task 1 数据集
- **类别**: bottle, cable, leather, pill, tile, transistor
- **图像数量**: 600张（每类100张）
- **图像格式**: PNG
- **用途**: 无监督聚类分析

### Task 3 数据集
- **记录数**: 26,200条
- **时间间隔**: 每10分钟一条记录
- **特征数**: 21个气象特征
- **预测目标**: 温度 (T in degC)
- **数据来源**: 德国气象站

## 模型性能

### Task 1 聚类性能
| 指标 | 数值 | 说明 |
|------|------|------|
| ARI | 1.0000 | 完美聚类 |
| NMI | 1.0000 | 完美聚类 |

### Task 3 预测性能
| 指标 | 数值 | 说明 |
|------|------|------|
| RMSE | ~0.5°C | 均方根误差 |
| MAE | ~0.3°C | 平均绝对误差 |
| R² | >0.95 | 决定系数 |

## 常见问题

### Q1: 如何使用GPU加速？
**A**: 确保安装了CUDA版本的PyTorch/TensorFlow，代码会自动检测并使用GPU。

### Q2: Task 1图像数据放在哪里？
**A**: 将图像放在`Task1/dataset/`目录下，确保有600张PNG图像（001.png - 600.png）。

### Q3: Task 3的数据预处理为什么需要标准化？
**A**: 神经网络对输入数据尺度敏感，标准化可以加速收敛、提高精度、避免梯度问题。

### Q4: 如何调整LSTM模型的超参数？
**A**: 可以在notebook中修改以下参数：
- `sequence_length`: 时间窗口长度（默认12）
- `LSTM units`: LSTM单元数量（默认64和32）
- `dropout_rate`: Dropout比率（默认0.2-0.3）
- `learning_rate`: 学习率（默认0.001）
- `batch_size`: 批次大小（默认64）

## 项目亮点

1. **迁移学习**: 利用ImageNet预训练的ResNet50提取高质量图像特征
2. **完美聚类**: 深度特征+K-means实现ARI和NMI均为1.0的完美聚类
3. **LSTM架构**: 双层LSTM有效捕捉时间序列的长期依赖关系
4. **鲁棒性处理**: 完整的数据清洗、缺失值处理、异常值处理流程
5. **多维可视化**: 6个子图全面展示模型性能和预测结果
6. **早停机制**: 防止过拟合，自动选择最佳模型权重

## 参考资料

- [ResNet论文](https://arxiv.org/abs/1512.03385)
- [K-means聚类算法](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [LSTM原论文](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [时间序列预测教程](https://www.tensorflow.org/tutorials/structured_data/time_series)

## 许可证

MIT License
