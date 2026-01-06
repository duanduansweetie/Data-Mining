import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from PIL import Image
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

def save_anomaly_grid(images_info, out_path, title):
    if not images_info:
        return
    
    n = len(images_info)
    cols = 8
    rows = math.ceil(n / cols)
    fig_width = cols * 2.5
    fig_height = rows * 2.5
    
    try:
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (path, score) in enumerate(images_info):
            try:
                img = Image.open(path).convert('RGB')
                axes[i].imshow(img)
                axes[i].set_title(f'{score:.2f}', fontsize=9)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, "Error", ha='center')
                axes[i].axis('off')
        for i in range(n, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        plt.close()

def analyze_scores(out_dir, manual_threshold=None, target_percentile=75, top_k=None):
    if not os.path.exists(out_dir):
     return
    csv_files = [f for f in os.listdir(out_dir) if f.endswith('_scores.csv')]
    
    if not csv_files:
        return

    for csv_file in csv_files:
        class_name = csv_file.replace('_scores.csv', '')
        file_path = os.path.join(out_dir, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            scores = df['score'].values
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            print(f"类别: {class_name}")
            print(f"样本数: {len(scores)}")
            print(f"分数范围: {min_score:.2f} ~ {max_score:.2f}")
            print(f"均值: {mean_score:.2f}, 标准差: {std_score:.2f}")
            
            threshold_3sigma = mean_score + 3 * std_score
            threshold_percentile = np.percentile(scores, target_percentile)
            
            print(f"参考阈值 {threshold_3sigma:.2f}")
            print(f"参考阈值 ({target_percentile}%分位数): {threshold_percentile:.2f}")
            if top_k is not None:
                anomalies = df.nlargest(top_k, 'score')
                if not anomalies.empty:
                    current_threshold = anomalies['score'].min()
                    thresh_desc = f"Top {top_k} (Min Score = {current_threshold:.2f})"
                else:
                    current_threshold = max_score
                    thresh_desc = f"Top {top_k} (无数据)"
            elif manual_threshold is not None:
                current_threshold = manual_threshold
                thresh_desc = f"手动指定 ({current_threshold})"
                anomalies = df[df['score'] >= current_threshold]
            else:
                current_threshold = threshold_percentile
                thresh_desc = f"自动 ({target_percentile}% 分位数 = {current_threshold:.2f})"
                anomalies = df[df['score'] >= current_threshold]

            count = len(anomalies)

            if count > 0:
                anomaly_list = []
                for _, row in anomalies.iterrows():
                    anomaly_list.append((row['path'], row['score']))
                anomaly_list.sort(key=lambda x: x[1], reverse=True)
                batch_size = 64
                num_pages = math.ceil(count / batch_size)
                
                for page in range(num_pages):
                    batch = anomaly_list[page*batch_size : (page+1)*batch_size]
                    save_name = f'{class_name}_anomalies_p{page+1}.png'
                    save_path = os.path.join(out_dir, save_name)
                    title = f"{class_name} - 监测到可能的异常图像 ({thresh_desc}) - Page {page+1}/{num_pages}"
                    save_anomaly_grid(batch, save_path, title)
            else:
                print("  没有样本超过该阈值。")
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(threshold_3sigma, color='r', linestyle='dashed', linewidth=1, label=f'Mean+3Std ({threshold_3sigma:.1f})')
            plt.axvline(threshold_percentile, color='g', linestyle='dashed', linewidth=1, label=f'{target_percentile}% Percentile ({threshold_percentile:.1f})')
            if top_k is not None:
                 plt.axvline(current_threshold, color='purple', linestyle='solid', linewidth=2, label=f'Top {top_k} Cutoff')
            elif manual_threshold is not None:
                plt.axvline(manual_threshold, color='purple', linestyle='solid', linewidth=2, label=f'Manual ({manual_threshold})')
            
            plt.title(f'Anomaly Score Distribution: {class_name}')
            plt.xlabel('Mahalanobis Score (Squared)')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(axis='y', alpha=0.5)
            
            save_path = os.path.join(out_dir, f'{class_name}_dist.png')
            plt.savefig(save_path)
            print(f"分布图已保存: {save_path}\n")
            plt.close()
            
        except Exception as e:
            print(f"分析 {csv_file} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=r"E:\programming_exercise\python\2\dataset", help='结果输出目录')
    parser.add_argument('--threshold', type=float, default=None, help='手动指定异常阈值 (优先级最高)')
    parser.add_argument('--percentile', type=float, default=75, help='指定百分位阈值 (默认80，即展示分数最高的前20%)')
    parser.add_argument('--top_k', type=int, default=None, help='直接指定展示分数最高的 K 张图片 (优先级高于百分位)')
    args = parser.parse_args()
    
    analyze_scores(args.out_dir, args.threshold, args.percentile, args.top_k)
