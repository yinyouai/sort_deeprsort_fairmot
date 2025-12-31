# utils.py
import numpy as np
import os
import glob

def load_kitti_dets(det_root):
    """
    det_root = data/kitti/training/det
    返回结构:
    {
        "0000": {0: array([...]), 1: array([...]), ...},
        "0001": {...}
    }
    """
    seqs = {}
    seq_dirs = sorted(glob.glob(os.path.join(det_root, "*")))

    for seq_dir in seq_dirs:
        seq_id = os.path.basename(seq_dir)
        seqs[seq_id] = {}

        det_files = sorted(glob.glob(os.path.join(seq_dir, "*.txt")))

        for f in det_files:
            frame = int(os.path.basename(f).split(".")[0])
            dets = []
            for line in open(f):
                vals = line.strip().split()
                x1, y1, x2, y2, score = map(float, vals[:5])
                dets.append([x1, y1, x2, y2])
            seqs[seq_id][frame] = np.array(dets)

    return seqs
