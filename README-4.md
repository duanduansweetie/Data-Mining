# Task 4: 甲状腺疾病异常检测

本项目使用 **孤立森林** 算法在仅含有正常样本的训练集上进行训练，并检测测试集中的异常（患病）样本。

## 快速运行

为了确保代码能够正常运行，请按照以下步骤配置环境：

### 1. 安装依赖项

确保您的电脑上已安装 Python（建议版本 3.8+）。在终端（Terminal）中运行以下命令安装必要的库：

```powershell
pip install -r requirements.txt
```

`requirements.txt` 中包含以下主要依赖：
- `pandas`, `numpy`: 数据处理
- `matplotlib`, `seaborn`: 数据可视化
- `scikit-learn`: 机器学习模型（孤立森林）
- `ipykernel`: Jupyter Notebook 运行内核

### 2. 运行 Notebook

您可以选择以下两种方式之一运行：

#### 方式 A：在 VS Code 中运行（推荐）
1. 在 VS Code 中直接打开 `result.ipynb`。
2. 如果提示选择内核，请选择已安装上述依赖的 Python 环境。
3. 点击顶部的 **"全部运行" (Run All)** 按钮。

#### 方式 B：使用 Jupyter Notebook 运行
在终端中输入以下命令：
```powershell
jupyter notebook result.ipynb
```

### 3. 数据说明
项目运行需要以下两个数据文件（已包含在仓库中）：
- `train-set.csv`: 训练集（仅包含正常样本）。
- `test-set.csv`: 测试集（包含正常和异常样本，用于评估）。

## 主要步骤
1. **数据导入与分析**：加载数据并可视化特征分布。
2. **数据预处理**：使用 `StandardScaler` 进行特征标准化。
3. **模型训练**：使用孤立森林学习正常数据的统计分布。
4. **结果评估**：计算 AUC-ROC、绘制混淆矩阵并生成分类报告。
