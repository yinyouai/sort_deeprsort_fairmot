import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_summary_csvs(metrics_dir, model_name):
    """加载所有评估结果并返回合并后的 DataFrame"""
    all_metrics = []
    for file in os.listdir(metrics_dir):
        if file.endswith(f"_{model_name}_metrics.csv"):
            file_path = os.path.join(metrics_dir, file)
            df = pd.read_csv(file_path)
            df['Sequence'] = file.split('_')[0]  # 提取序列ID
            df['Frame'] = df.index  # 使用行号作为 Frame
            all_metrics.append(df)

    # 合并所有 DataFrame 并返回
    combined_df = pd.concat(all_metrics, axis=0)
    return combined_df


def visualize_metrics(metrics_df, output_dir):
    """
    可视化 MOTA, MOTP, IDF1 等评估指标
    """
    # 确保没有缺失值
    metrics_df = metrics_df.dropna(subset=['mota', 'motp', 'idf1', 'precision', 'recall'])

    metrics = ['mota', 'motp', 'idf1', 'precision', 'recall']
    plt.figure(figsize=(12, 8))

    for idx, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, idx)  # 设置 2 行 3 列的子图布局
        sns.lineplot(data=metrics_df, x="Frame", y=metric, hue="Sequence")  # 删除 marker 点，使用默认连线
        plt.title(f"{metric.upper()} per Sequence")
        plt.xlabel("Frame")
        plt.ylabel(metric.upper())

    # 保存可视化图像
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Metrics visualization saved to: {output_path}")


def main():
    # 配置目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # analysis/
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    metrics_dir = os.path.join(PROJECT_ROOT, "results_analysis", "metrics")
    model_name = 'sort'  # 可以替换为其他模型名，如 'deepsort'
    output_dir = os.path.join(PROJECT_ROOT, "results_analysis", "vis", model_name)

    # 加载所有序列的评估结果
    metrics_df = load_summary_csvs(metrics_dir, model_name)

    # 检查加载的数据
    print(f"Loaded metrics data:\n{metrics_df.head()}")

    # 可视化评估指标
    visualize_metrics(metrics_df, output_dir)


if __name__ == "__main__":
    main()
