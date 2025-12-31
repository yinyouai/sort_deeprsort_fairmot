import os
import numpy as np
import pandas as pd
import argparse

def load_tracker_file(file_path):
    """
    读取 tracker 文件
    格式假设：frame_id track_id x1 y1 x2 y2
    返回：np.array, shape=(N,6)
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            cols = line.strip().split()
            if len(cols) < 6:
                continue
            frame = int(cols[0])
            tid = int(cols[1])
            x1, y1, x2, y2 = map(float, cols[2:6])
            data.append([frame, tid, x1, y1, x2, y2])
    return np.array(data).reshape(-1,6)

def compute_statistics(trk_data):
    """
    计算统计指标
    trk_data: np.array, shape=(N,6)
    """
    stats = {}

    if trk_data.size == 0:
        stats['num_tracks'] = 0
        stats['avg_length'] = 0
        stats['max_length'] = 0
        stats['num_fragments'] = 0
        stats['avg_speed'] = 0
        stats['avg_acceleration'] = 0
        return stats

    track_ids = np.unique(trk_data[:,1])
    stats['num_tracks'] = len(track_ids)

    lengths = []
    fragments = []
    speeds = []
    accelerations = []

    for tid in track_ids:
        frames = trk_data[trk_data[:,1]==tid][:,0]
        frames_sorted = np.sort(frames)
        length = len(frames_sorted)
        lengths.append(length)

        diffs = np.diff(frames_sorted)
        num_frag = np.sum(diffs>1) + 1
        fragments.append(num_frag)

        coords = trk_data[trk_data[:,1]==tid][:,2:4]  # 使用左上角
        if len(coords) > 1:
            delta = np.diff(coords, axis=0)
            spd = np.linalg.norm(delta, axis=1)
            speeds.extend(spd)

            if len(spd) > 1:
                acc = np.diff(spd)
                accelerations.extend(acc)

    stats['avg_length'] = np.mean(lengths)
    stats['max_length'] = np.max(lengths)
    stats['num_fragments'] = np.mean(fragments)
    stats['avg_speed'] = np.mean(speeds) if speeds else 0
    stats['avg_acceleration'] = np.mean(accelerations) if accelerations else 0

    return stats

def main():
    # 获取当前脚本所在目录: .../work/analysis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (假设是上一级): .../work
    project_root = os.path.dirname(current_dir)
    
    # 构建默认的 tracker_root 路径: .../work/results/deepsort_result
    default_tracker_root = os.path.join(project_root, 'results', 'deepsort_result')
    
    # 构建默认的 save_dir 路径: .../work/results_analysis/deepsort_metrics
    default_save_dir = os.path.join(project_root, 'results_analysis', 'deepsort_metrics')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_root', type=str, default=default_tracker_root, help='Tracker 输出文件夹')
    parser.add_argument('--save_dir', type=str, default=default_save_dir, help='保存 CSV 文件夹')
    args = parser.parse_args()

    print(f"Tracker root: {args.tracker_root}")
    print(f"Save dir: {args.save_dir}")
    
    if not os.path.exists(args.tracker_root):
        print(f"Error: Tracker root path does not exist: {args.tracker_root}")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    tracker_files = [f for f in os.listdir(args.tracker_root) if f.endswith('.txt')]
    tracker_files.sort()

    if not tracker_files:
        print(f"Warning: No .txt files found in {args.tracker_root}")

    all_stats = []

    for trk_file in tracker_files:
        seq_id = os.path.splitext(trk_file)[0]
        trk_path = os.path.join(args.tracker_root, trk_file)
        trk_data = load_tracker_file(trk_path)
        stats = compute_statistics(trk_data)
        stats['sequence'] = seq_id
        all_stats.append(stats)

        # 保存每个序列的 CSV
        df_seq = pd.DataFrame([stats])
        df_seq = df_seq[['sequence','num_tracks','avg_length','max_length','num_fragments','avg_speed','avg_acceleration']]
        df_seq.to_csv(os.path.join(args.save_dir, f"{seq_id}_stats.csv"), index=False)

    if all_stats:
        # 保存汇总 CSV
        df_all = pd.DataFrame(all_stats)
        df_all = df_all[['sequence','num_tracks','avg_length','max_length','num_fragments','avg_speed','avg_acceleration']]
        df_all.to_csv(os.path.join(args.save_dir, "summary_stats.csv"), index=False)
        print(f"统计分析完成，结果保存至 {args.save_dir}")
    else:
        print("未处理任何文件。")

if __name__ == "__main__":
    main()
