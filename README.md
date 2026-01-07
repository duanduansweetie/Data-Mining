# 多模态异常检测框架

统一的异常检测框架，支持图像和表格数据。

## 核心文件

- `Data-Mining-task-2/multimodal_anomaly_detection.py` - 统一框架（必须有）
- `Data-Mining-task-2/anomaly_detection.py` - Task2 图像异常检测
- `Data-Mining-task-2/analyze_results.py` - 分析工具
- `Data-Mining-task-4/anomaly_detection_tabular.py` - Task4 表格异常检测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### Task2 - 图像异常检测

```bash
cd Data-Mining-task-2
python anomaly_detection.py --data_root ./Image_Anomaly_Detection
```

可选参数：
```bash
--method isolation_forest  # 使用孤立森林
--method one_class_svm     # 使用One-Class SVM
--device cpu               # 强制使用CPU
```

### Task4 - 表格异常检测

```bash
cd Data-Mining-task-4
python anomaly_detection_tabular.py --compare
```

可选参数：
```bash
--train_csv <path>      # 训练集路径（默认：./train-set.csv）
--test_csv <path>       # 测试集路径（默认：./test-set.csv）
--method isolation_forest  # 使用孤立森林
--method mahalanobis    # 使用马氏距离
--method one_class_svm  # 使用One-Class SVM
--compare               # 比较所有算法
--skip_viz              # 跳过可视化
```

## 输出说明

### 文件结构
```
dataset/
├── hazelnut_scores.csv      # 异常分数文件
├── hazelnut_top_anom.png    # 最异常的图像可视化
├── zipper_scores.csv
└── zipper_top_anom.png
```

### CSV文件格式
```csv
path,label,score
"图片路径",0,123.456789
```
- **label**: -1表示未知（正常）
- **score**: 异常分数（越大越异常）

### 关于标签和分数

**Q: 为什么标签都是-1（unknown）？**
- 这是正常的！异常检测往往没有测试标签
- test目录下直接是图片，没有分类
- 需要根据分数大小来判断异常

**Q: 异常分数很大/很小？**
- 分数的绝对值不重要
- 重要的是相对大小：分数越大的越可能是异常
- 马氏距离衡量与正常分布的偏离程度

**Q: 如何确定哪些是异常？**
1. 查看 `{类别}_top_anom.png` - 显示分数最高的8张图
2. 查看 `{类别}_scores.csv` - 按score排序
3. 选择前K张或分数超过阈值的作为异常

```python
import pandas as pd
df = pd.read_csv('dataset/hazelnut_scores.csv')
top_anomalies = df.nlargest(10, 'score')  # 最异常的10张
print(top_anomalies)
```

## 参数说明

### Task2
- `--data_root`: 数据目录（必填）
- `--out_dir`: 输出目录（默认：./dataset）
- `--method`: 算法（mahalanobis/isolation_forest/one_class_svm）
- `--device`: 设备（cuda/cpu）

### Task4
- `--train_csv`: 训练集路径
- `--test_csv`: 测试集路径
- `--method`: 算法
- `--compare`: 比较所有算法
- `--skip_viz`: 跳过可视化

## 工作原理

### Task2（图像）
1. **训练**：只用good样本训练，学习正常图像的特征分布
2. **检测**：计算测试图像与正常分布的马氏距离
3. **输出**：距离越大 = 越异常

### Task4（表格）
1. **训练**：只用正常样本训练孤立森林
2. **检测**：计算异常分数
3. **输出**：支持多种算法自动比较

## 特性

- 统一接口：图像和表格使用相同API
- 多算法：马氏距离、孤立森林、One-Class SVM
- 自动化：Task4可自动比较所有算法
- 易扩展：模块化设计

## 常见问题

**Q: CUDA错误？**
```bash
python anomaly_detection.py --data_root ./Image_Anomaly_Detection --device cpu
```

**Q: 找不到数据？**
```bash
ls ./Image_Anomaly_Detection
```

**Q: 如何提高准确率？**
- 尝试不同的算法（`--method`）
- 增加训练样本数量
- 调整阈值（根据分数分布）
