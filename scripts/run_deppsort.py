import os
import sys
import cv2
import numpy as np

# ================= 路径设置 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from deepsort.deepsort import DeepSort

DET_ROOT = os.path.join(PROJECT_ROOT, "data/kitti/training/det")
IMG_ROOT = os.path.join(PROJECT_ROOT, "data/kitti/training/image_02")
OUT_ROOT = os.path.join(PROJECT_ROOT, "results/deepsort_result")

os.makedirs(OUT_ROOT, exist_ok=True)


def load_frame_dets(det_file):
    """
    每个 det txt 只对应一帧
    格式假设：
    x1 y1 x2 y2 score
    或
    x1 y1 x2 y2
    """
    boxes = []
    scores = []

    with open(det_file, "r") as f:
        for line in f:
            items = list(map(float, line.strip().split()))
            boxes.append(items[:4])
            scores.append(items[4] if len(items) > 4 else 1.0)

    return np.array(boxes), np.array(scores)


def run_sequence(seq):
    print(f"Running DeepSORT + CNN ReID on sequence {seq}")

    det_seq_dir = os.path.join(DET_ROOT, seq)
    img_seq_dir = os.path.join(IMG_ROOT, seq)



    tracker = DeepSort(
        model_path=os.path.join(PROJECT_ROOT, "src/deepsort/reid/ckpt.t7"),
        max_age=30,
        n_init=3
    )

    out_path = os.path.join(OUT_ROOT, f"{seq}.txt")
    out_file = open(out_path, "w")

    det_files = sorted(os.listdir(det_seq_dir))

    for det_name in det_files:
        frame_id = int(os.path.splitext(det_name)[0])
        det_path = os.path.join(det_seq_dir, det_name)

        img_path = os.path.join(img_seq_dir, f"{frame_id:06d}.png")
        # 使用numpy读取以支持中文路径
        img_array = np.fromfile(img_path, dtype=np.uint8)
        if len(img_array) == 0:
            print(f"[WARN] Missing image: {img_path}")
            continue
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Failed to decode image: {img_path}")
            continue

        boxes, scores = load_frame_dets(det_path)
        if len(boxes) == 0:
            continue

        outputs = tracker.update(boxes, scores, image)

        for track in outputs:
            tid, x1, y1, x2, y2 = track
            out_file.write(
                f"{frame_id} {tid} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n"
            )

    out_file.close()
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    if not os.path.exists(IMG_ROOT):
        raise FileNotFoundError(f"IMG_ROOT not found: {IMG_ROOT}")
    
    sequences = sorted(os.listdir(IMG_ROOT))
    print(f"Found {len(sequences)} sequences: {sequences}")
    
    for seq in sequences:
        run_sequence(seq)
    
    print("\nAll DeepSORT tracking completed.")
    print(f"Results saved to: {OUT_ROOT}")
