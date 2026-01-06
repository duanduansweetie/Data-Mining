# 图像异常检测基线模型 (Image Anomaly Detection Baseline)

基于 ResNet-50 深度特征嵌入与马氏距离 (Mahalanobis Distance) 的无监督异常检测实现。

## 项目结构 (Project Structure)
- `anomaly_detection.py`: 主脚本，包含特征提取、高斯分布拟合与异常分数预测。
- `analyze_results.py`: 分析工具，用于计算统计量、确定阈值及生成可视化图表。
- `design_document.md`: 详细的实验报告，包含算法原理、训练过程与效果评估。

## 使用说明 (Usage)

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

2. 运行异常检测:
   ```bash
   python anomaly_detection.py --data_root ./Image_Anomaly_Detection --out_dir ./dataset
   ```

3. 分析结果与可视化:
   ```bash
   python analyze_results.py
   ```
