# 文件路径: analysis/eval_compare_kitti.py
# 功能:
# 1) 将 KITTI label_02 转为 MOT GT
# 2) 将 SORT / DeepSORT / FairMOT 输出转 MOT
# 3) 计算 MOT 指标
# 输出: results_analysis/compare/mot_metrics_comparison.csv

import os
import pandas as pd
import motmetrics as mm
import tempfile

# ================== 基础路径 ==================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# KITTI Tracking GT
kitti_label_folder = os.path.join(
    project_root, 'data', 'kitti', 'training', 'label_02'
)

# GT 转换后的 MOT 格式
mot_gt_folder = os.path.join(
    project_root, 'results_analysis', 'gt_mot_format'
)
os.makedirs(mot_gt_folder, exist_ok=True)

# 跟踪结果（你已有）
results_folders = {
    'SORT': os.path.join(project_root, 'results', 'sort_results'),
    'DeepSORT': os.path.join(project_root, 'results', 'deepsort_result'),
    'FairMOT': os.path.join(project_root, 'results', 'fairmot_result')
}

# 指标输出
output_dir = os.path.join(project_root, 'results_analysis', 'compare')
os.makedirs(output_dir, exist_ok=True)

# ================== Step 1: label_02 -> MOT ==================
print("=== Step 1: Convert KITTI label_02 to MOT GT ===")

VALID_CLASSES = {'Car'}  # 可改成 {'Car', 'Pedestrian', 'Cyclist'}

for seq_file in sorted(os.listdir(kitti_label_folder)):
    if not seq_file.endswith('.txt'):
        continue

    seq_name = seq_file.replace('.txt', '')
    src_path = os.path.join(kitti_label_folder, seq_file)
    dst_path = os.path.join(mot_gt_folder, seq_file)

    mot_lines = []

    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue

            frame = int(parts[0]) + 1          # MOT 从 1 开始
            track_id = int(parts[1]) + 1       # MOT ID 从 1 开始
            cls = parts[2]

            if cls not in VALID_CLASSES:
                continue

            x1, y1, x2, y2 = map(float, parts[6:10])
            w = x2 - x1
            h = y2 - y1

            mot_lines.append([
                frame, track_id,
                x1, y1, w, h,
                1, -1, -1, -1
            ])

    if mot_lines:
        pd.DataFrame(mot_lines).to_csv(dst_path, header=False, index=False)
        print(f"{seq_name} -> 转换完成")
    else:
        print(f"{seq_name} -> 无有效 GT，跳过")
# -------------------- 2. 输出转换为 MOT 格式 --------------------
def convert_to_mot_format(file_path):
    """
    将 SORT/DeepSORT/FairMOT 输出转换为 MOT 格式
    MOT 格式: frame, id, x, y, w, h, conf, -1, -1, -1
    """
    mot_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                frame_id = int(float(parts[0])) + 1  # frame 从1开始
                track_id = int(float(parts[1])) + 1  # track id 从1开始
                x1, y1, x2, y2 = map(float, parts[2:6])
                w = x2 - x1
                h = y2 - y1
                mot_line = [frame_id, track_id, x1, y1, w, h, 1, -1, -1, -1]
                mot_lines.append(mot_line)
            except ValueError:
                continue
    return mot_lines

# ================== Step 2: 结果转 MOT ==================
def convert_tracker_to_mot(txt_path):
    """
    统一 SORT / DeepSORT / FairMOT 输出
    格式假设: frame id x1 y1 x2 y2
    """
    mot = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            frame = int(float(parts[0])) + 1
            tid = int(float(parts[1])) + 1
            x1, y1, x2, y2 = map(float, parts[2:6])

            mot.append([
                frame, tid,
                x1, y1, x2 - x1, y2 - y1,
                1, -1, -1, -1
            ])
    return mot

# ================== Step 3: 评测 ==================
def evaluate(gt_folder, result_folder):
    accs = []
    names = []

    for gt_file in sorted(os.listdir(gt_folder)):
        if not gt_file.endswith('.txt'):
            continue

        seq = gt_file.replace('.txt', '')
        gt_path = os.path.join(gt_folder, gt_file)
        res_path = os.path.join(result_folder, f'{seq}.txt')

        if not os.path.exists(res_path):
            print(f'{seq} -> 无结果文件，跳过')
            continue

        try:
            gt = mm.io.loadtxt(gt_path, fmt='mot15-2D')

            mot_lines = convert_to_mot_format(res_path)
            if len(mot_lines) == 0:
                print(f'{seq} -> 结果为空，跳过')
                continue

            df = pd.DataFrame(mot_lines)
            tmp_path = os.path.join(output_dir, f'_tmp_{seq}.txt')
            df.to_csv(tmp_path, header=False, index=False)

            ts = mm.io.loadtxt(tmp_path, fmt='mot15-2D')

            # ✅ 关键修复在这一行
            acc = mm.utils.compare_to_groundtruth(
                gt, ts, 'iou', distth=0.5
            )

            accs.append(acc)
            names.append(seq)

        except Exception as e:
            print(f'{seq} -> 出错: {e}')

    return accs, names


# ================== Step 4: 汇总 ==================
mh = mm.metrics.create()
all_results = {}

metrics = [
    'num_frames', 'mota', 'motp', 'idf1',
    'num_switches', 'num_false_positives', 'num_misses'
]

for method, folder in results_folders.items():
    print(f"\nEvaluating {method} ...")
    accs, names = evaluate(mot_gt_folder, folder)

    if not accs:
        print(f"{method}: 无有效结果")
        continue

    summary = mh.compute_many(
        accs, names=names,
        metrics=metrics,
        generate_overall=True
    )

    summary = summary.rename(columns={
        'num_frames': 'Frames',
        'num_switches': 'IDSW',
        'num_false_positives': 'FP',
        'num_misses': 'FN'
    })

    all_results[method] = summary
    print(summary)

# ================== 保存 ==================
if all_results:
    final = pd.concat(all_results.values(), keys=all_results.keys())
    out_csv = os.path.join(output_dir, 'mot_metrics_comparison.csv')
    final.to_csv(out_csv)
    print(f"\n✅ 指标已保存到: {out_csv}")
else:
    print("\n❌ 没有生成任何指标")
