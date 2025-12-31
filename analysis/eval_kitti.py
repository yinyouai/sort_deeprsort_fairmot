import os
import numpy as np
import pandas as pd
import motmetrics as mm
import argparse

def load_tracking(file_path, is_gt=True):
    """
    读取 tracking 文件
    GT 文件 (KITTI label_02):
        frame_id track_id ... bbox_left bbox_top bbox_right bbox_bottom ...
    Tracking 文件 (SORT/DeepSORT):
        frame_id track_id x1 y1 x2 y2
    返回: np.array, shape = (N,6)
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            cols = line.strip().split()
            try:
                if is_gt:
                    if len(cols) < 10:
                        continue
                    frame = int(cols[0])
                    tid = int(cols[1])
                    x1, y1, x2, y2 = map(float, cols[6:10])
                else:
                    if len(cols) < 6:
                        continue
                    frame = int(cols[0])
                    tid = int(cols[1])
                    x1, y1, x2, y2 = map(float, cols[2:6])
                data.append([frame, tid, x1, y1, x2, y2])
            except:
                continue
    return np.array(data).reshape(-1,6)

def eval_sequence(gt_file, trk_file):
    """
    使用 motmetrics 评估单序列
    返回: pandas.DataFrame
    """
    gt = load_tracking(gt_file, is_gt=True)
    trk = load_tracking(trk_file, is_gt=False)

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt[:,0]) | set(trk[:,0]))

    for f in frames:
        gt_ids = gt[gt[:,0]==f,1].astype(str)
        gt_boxes = gt[gt[:,0]==f,2:6]
        trk_ids = trk[trk[:,0]==f,1].astype(str)
        trk_boxes = trk[trk[:,0]==f,2:6]

        if len(gt_boxes)==0:
            dists = np.empty((0,len(trk_boxes)))
        elif len(trk_boxes)==0:
            dists = np.empty((len(gt_boxes),0))
        else:
            dists = mm.distances.iou_matrix(gt_boxes, trk_boxes, max_iou=0.5)

        acc.update(gt_ids, trk_ids, dists)

    mh = mm.metrics.create()
    METRICS = ['mota', 'motp', 'idf1', 'num_switches', 'num_objects', 'precision', 'recall']
    summary = mh.compute(acc, metrics=METRICS, name=os.path.basename(trk_file).split('.')[0])
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sort', help='实验模型名称，用于输出标记')
    args = parser.parse_args()

    # 动态获取项目根目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # analysis/
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    # tracker 文件夹
    trk_root = os.path.join(PROJECT_ROOT, "results", "sort_results")
    # KITTI GT 文件夹
    gt_root = os.path.join(PROJECT_ROOT, "data/kitti/training/label_02")
    # 输出目录
    save_dir = os.path.join(PROJECT_ROOT, "results_analysis", "metrics")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(trk_root):
        raise FileNotFoundError(f"Tracking results not found: {trk_root}")

    # 列出 tracker 文件
    seqs = sorted([f.split('.')[0] for f in os.listdir(trk_root) if f.endswith(".txt")])
    if not seqs:
        print(f"No tracker files found in {trk_root}")
        return

    all_summary = []

    for seq_id in seqs:
        gt_file = os.path.join(gt_root, f"{seq_id}.txt")
        trk_file = os.path.join(trk_root, f"{seq_id}.txt")
        if not os.path.exists(gt_file):
            print(f"Warning: GT file not found: {gt_file}")
            continue
        summary = eval_sequence(gt_file, trk_file)
        all_summary.append(summary)
        # 保存每序列 CSV
        summary.to_csv(os.path.join(save_dir, f"{seq_id}_{args.model}_metrics.csv"))

    # 汇总所有序列
    if all_summary:
        final = pd.concat(all_summary, axis=0)
        final.to_csv(os.path.join(save_dir, f"{args.model}_summary.csv"))
        print(f"Metrics saved to {save_dir}")
    else:
        print("No sequences were evaluated.")

if __name__ == "__main__":
    main()
