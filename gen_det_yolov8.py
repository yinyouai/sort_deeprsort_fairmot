import os
from ultralytics import YOLO
import cv2

# 指定 KITTI 目录
ROOT = "data/kitti/training"
IMG_ROOT = f"{ROOT}/image_02"
SAVE_ROOT = f"{ROOT}/det"

os.makedirs(SAVE_ROOT, exist_ok=True)

# 使用 YOLOv8n（推荐）
model = YOLO("yolov8n.pt")

def process_sequence(seq_id):
    seq_path = os.path.join(IMG_ROOT, seq_id)
    save_seq = os.path.join(SAVE_ROOT, seq_id)
    os.makedirs(save_seq, exist_ok=True)

    img_files = sorted(os.listdir(seq_path))

    for img_name in img_files:
        frame_id = int(img_name.split(".")[0])
        img_path = os.path.join(seq_path, img_name)
        img = cv2.imread(img_path)

        results = model(img)[0]

        det_file = os.path.join(save_seq, f"{frame_id:06d}.txt")
        with open(det_file, "w") as f:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                f.write(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.2f}\n")

    print(f"Sequence {seq_id} done!")


def main():
    seqs = sorted(os.listdir(IMG_ROOT))
    for seq_id in seqs:
        process_sequence(seq_id)

    print("\nAll sequences processed!")
    print(f"Det files saved to {SAVE_ROOT}")


if __name__ == "__main__":
    main()
