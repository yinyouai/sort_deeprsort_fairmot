import os
import sys
import glob
import numpy as np

# 将 project 根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fairmot.tracker import Tracker, Detection
from src.fairmot.fairmot_model import FairMOTModel

# -----------------------------
# 配置
# -----------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data/kitti/training")
DET_DIR = os.path.join(DATA_DIR, "det")   # YOLO 输出目录
RESULT_DIR = os.path.join(PROJECT_ROOT, "results/fairmot_result")
WEIGHT_PATH = None  # 如果有权重就填路径

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# YOLO 转换成 same detections
# -----------------------------
def read_yolo_frame_txt(file_path):
    """
    读取 YOLO 输出 txt，返回 Detection 对象列表
    YOLO txt 每行: x1 y1 x2 y2 score
    """
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2, score = map(float, parts[:5])
            detections.append(Detection(x1, y1, x2, y2, score))
    return detections

# -----------------------------
# 运行单个序列
# -----------------------------
def run_sequence(seq_name):
    seq_det_dir = os.path.join(DET_DIR, seq_name)
    frame_files = sorted(glob.glob(os.path.join(seq_det_dir, "*.txt")))
    if len(frame_files) == 0:
        print(f"[WARNING] No detection files found for sequence {seq_name}")
        return

    print(f"\n[INFO] Running sequence {seq_name}, {len(frame_files)} frames found")

    tracker = Tracker(max_age=30)
    model = FairMOTModel(weight_path=WEIGHT_PATH, embedding_dim=128)
    results = []

    for frame_file in frame_files:
        frame_id = int(os.path.basename(frame_file).split('.')[0])
        dets = read_yolo_frame_txt(frame_file)
        embeddings = model.extract_embedding(None, len(dets))
        online_targets = tracker.update(dets, embeddings)
        for t_id, bbox in online_targets:
            x1, y1, x2, y2 = bbox
            # same detections 格式: frame_id track_id x1 y1 x2 y2 score -1 -1 -1 -1
            results.append([frame_id, t_id, x1, y1, x2, y2, -1, -1, -1, -1, -1])

    out_file = os.path.join(RESULT_DIR, f"{seq_name}.txt")
    np.savetxt(out_file, results, fmt="%.2f")
    print(f"[INFO] Sequence {seq_name} finished. Results saved to {out_file}")

# -----------------------------
# 主函数
# -----------------------------
def main():
    if not os.path.exists(DET_DIR):
        print(f"[ERROR] Detection directory does not exist: {DET_DIR}")
        return

    sequences = sorted([d for d in os.listdir(DET_DIR) if os.path.isdir(os.path.join(DET_DIR, d))])
    print(f"[INFO] Found {len(sequences)} sequences in {DET_DIR}")

    if len(sequences) == 0:
        print("[WARNING] No sequences found. Check your detection folder!")
        return

    for seq in sequences:
        run_sequence(seq)

if __name__ == "__main__":
    main()
