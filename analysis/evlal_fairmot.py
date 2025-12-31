import os
import sys
import glob
import numpy as np
import pandas as pd

# -------------------------------------------------
# project root
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# 路径配置
# -------------------------------------------------
TRACK_RESULT_DIR = os.path.join(PROJECT_ROOT, "results/fairmot_result")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_analysis/fairmot_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# IoU
# -------------------------------------------------
def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-6)

# -------------------------------------------------
# bbox center distance
# -------------------------------------------------
def center_dist(box, boxes):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    cys = (boxes[:, 1] + boxes[:, 3]) / 2
    return np.sqrt((cxs - cx) ** 2 + (cys - cy) ** 2)

# -------------------------------------------------
# 读取 MOT txt
# -------------------------------------------------
def load_mot_result(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
        names=["frame","track_id","x1","y1","x2","y2",
               "score","a","b","c","d"]
    )
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df

# -------------------------------------------------
# 单序列分析（优化版）
# -------------------------------------------------
def analyze_sequence(df, seq_name):
    metrics = []

    # 1. 按 frame 预分组，加速查找同帧物体
    # 提取 numpy 数组以加速后续计算
    frame_groups = {
        f: g[["x1","y1","x2","y2"]].values
        for f, g in df.groupby("frame")
    }

    MAX_DIST = 200  # 像素阈值（只计算近距离遮挡）
    SAMPLE_FRAME_STEP = 1  # 可改为 5 做稀疏采样

    # 2. 按 track_id 预分组，避免在循环中反复全表扫描 (关键性能优化)
    track_groups = df.groupby("track_id")
    total_tracks = len(track_groups)
    print(f"  [DEBUG] Processing {total_tracks} tracks in {seq_name}...")

    for i, (tid, track) in enumerate(track_groups):
        # 简单的进度打印
        if i % 50 == 0:
            print(f"  [DEBUG] Processed {i}/{total_tracks} tracks...", end='\r')

        # 确保按帧排序
        track = track.sort_values("frame")
        
        frames = track["frame"].values
        boxes = track[["x1","y1","x2","y2"]].values

        track_len = len(frames)
        if track_len < 2:
            continue

        gaps = np.diff(frames) - 1
        num_gaps = int(np.sum(gaps > 0))
        max_gap = int(np.max(gaps[gaps > 0])) if num_gaps > 0 else 0
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])

        occlusions = []
        
        # 遍历该轨迹的每一帧 (或采样)
        sample_indices = range(0, track_len, SAMPLE_FRAME_STEP)
        
        for idx in sample_indices:
            f = frames[idx]
            box = boxes[idx]
            
            same_frame = frame_groups[f]
            if len(same_frame) <= 1:
                occlusions.append(0.0)
                continue

            # 距离筛选
            dists = center_dist(box, same_frame)
            
            # 筛选距离小于阈值
            mask = dists < MAX_DIST
            near_boxes = same_frame[mask]

            if len(near_boxes) <= 1:
                occlusions.append(0.0)
                continue

            # 排除自身 (坐标完全相同)
            is_self = np.all(np.isclose(near_boxes, box, atol=1e-3), axis=1)
            others = near_boxes[~is_self]
            
            if len(others) == 0:
                occlusions.append(0.0)
            else:
                occlusions.append(np.max(compute_iou(box, others)))

        metrics.append({
            "sequence": seq_name,
            "track_id": int(tid),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "track_length": int(track_len),
            "num_gaps": num_gaps,
            "max_gap": max_gap,
            "mean_area": float(np.mean(areas)),
            "max_occlusion_iou": float(np.max(occlusions)) if occlusions else 0.0,
            "mean_occlusion_iou": float(np.mean(occlusions)) if occlusions else 0.0
        })

    print(f"  [DEBUG] Finished processing {total_tracks} tracks.          ")
    return pd.DataFrame(metrics)

# -------------------------------------------------
# 主函数
# -------------------------------------------------
def main():
    seq_files = sorted(glob.glob(os.path.join(TRACK_RESULT_DIR, "*.txt")))
    print(f"[INFO] Found {len(seq_files)} FairMOT sequences")

    all_metrics = []

    for seq_path in seq_files:
        seq_name = os.path.splitext(os.path.basename(seq_path))[0]
        print(f"[INFO] Evaluating FairMOT sequence {seq_name}")

        df = load_mot_result(seq_path)
        metrics_df = analyze_sequence(df, seq_name)

        out_csv = os.path.join(OUT_DIR, f"{seq_name}_metrics.csv")
        metrics_df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved {out_csv}")

        all_metrics.append(metrics_df)

    if len(all_metrics) > 0:
        all_df = pd.concat(all_metrics, ignore_index=True)
        all_out = os.path.join(OUT_DIR, "fairmot_all_metrics.csv")
        all_df.to_csv(all_out, index=False)
        print(f"[INFO] Saved merged metrics: {all_out}")

if __name__ == "__main__":
    main()
