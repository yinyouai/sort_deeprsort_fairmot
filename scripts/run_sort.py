import os
import numpy as np
from src.sort.sort import SORT

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# KITTI 路径
ROOT = os.path.join(PROJECT_ROOT, "data/kitti/training")
IMG_ROOT = os.path.join(ROOT, "image_02")
DET_ROOT = os.path.join(ROOT, "det")
SAVE_ROOT = os.path.join(PROJECT_ROOT, "results/sort_results")
os.makedirs(SAVE_ROOT, exist_ok=True)


def load_detections(det_file):
    """
    读取 YOLOv8 生成的 detections
    格式：x1 y1 x2 y2 score
    """
    dets = []
    if os.path.exists(det_file):
        with open(det_file, "r") as f:
            for line in f.readlines():
                x1, y1, x2, y2, score = map(float, line.strip().split())
                dets.append([x1, y1, x2, y2, score])
    return np.array(dets) if dets else np.empty((0,5))


def process_sequence(seq_id):
    seq_img_dir = os.path.join(IMG_ROOT, seq_id)
    seq_det_dir = os.path.join(DET_ROOT, seq_id)
    seq_save_file = os.path.join(SAVE_ROOT, f"{seq_id}.txt")

    img_files = sorted(os.listdir(seq_img_dir))
    tracker = SORT()

    with open(seq_save_file, "w") as out:
        for img_name in img_files:
            frame_id = int(img_name.split(".")[0])
            det_file = os.path.join(seq_det_dir, f"{frame_id:06d}.txt")

            dets = load_detections(det_file)

            # 过滤异常框，去掉宽高 <1 的框
            if dets.size > 0:
                valid_mask = (dets[:,2] - dets[:,0] > 1) & (dets[:,3] - dets[:,1] > 1)
                valid_dets = dets[valid_mask, :4]
                if valid_dets.size > 0:
                    track_bbs_ids = tracker.update(valid_dets)
                else:
                    track_bbs_ids = tracker.update(np.empty((0,4)))
            else:
                track_bbs_ids = tracker.update(np.empty((0,4)))

            # 写 KITTI 输出格式
            for d in track_bbs_ids:
                x1, y1, x2, y2, track_id = d
                out.write(f"{frame_id} {int(track_id)} {x1:.2f} {y1:.2f} "
                          f"{x2:.2f} {y2:.2f}\n")

    print(f"Sequence {seq_id} tracking done!")


def main():
    if not os.path.exists(IMG_ROOT):
        raise FileNotFoundError(f"IMG_ROOT not found: {IMG_ROOT}")

    seqs = sorted(os.listdir(IMG_ROOT))
    print(f"Found {len(seqs)} sequences: {seqs}")

    for seq_id in seqs:
        process_sequence(seq_id)

    print("\nAll SORT tracking completed.")
    print(f"Results saved to: {SAVE_ROOT}")


if __name__ == "__main__":
    main()
