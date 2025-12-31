# sort/sort.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanFilter

def iou(b1, b2):
    """bbox: x1,y1,x2,y2"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    area2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    if area1 + area2 - inter == 0:
        return 0.0
    return inter / float(area1 + area2 - inter)

class Track:
    def __init__(self, mean, cov, track_id):
        self.mean = mean
        self.cov = cov
        self.track_id = track_id
        self.age = 0
        self.time_since_update = 0

    def to_bbox(self):
        cx, cy, s, r = self.mean[:4]
        w = np.sqrt(s * r)
        h = s / max(w,1e-6)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

class SORT:
    def __init__(self, max_age=3, min_hits=3, iou_thresh=0.3):
        self.tracker = KalmanFilter()
        self.tracks = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.track_id = 0

    def update(self, detections):
        self.frame_count += 1

        # 预测现有轨迹
        for t in self.tracks:
            t.mean, t.cov = self.tracker.predict(t.mean, t.cov)

        if len(detections) == 0:
            # 只更新时间
            for t in self.tracks:
                t.time_since_update += 1
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return []

        # 生成预测 bbox
        predicted_boxes = [t.to_bbox() for t in self.tracks]
        iou_matrix = np.zeros((len(predicted_boxes), len(detections)))

        for i, p in enumerate(predicted_boxes):
            for j, det in enumerate(detections):
                iou_matrix[i,j] = iou(p, det)

        if iou_matrix.size > 0:
            row, col = linear_sum_assignment(-iou_matrix)
        else:
            row, col = np.array([]), np.array([])

        matched, unmatched_trk, unmatched_det = [], [], []
        for t, _ in enumerate(predicted_boxes):
            if t not in row:
                unmatched_trk.append(t)
        for d, _ in enumerate(detections):
            if d not in col:
                unmatched_det.append(d)

        # IOU 筛选
        for r, c in zip(row, col):
            if iou_matrix[r, c] < self.iou_thresh:
                unmatched_trk.append(r)
                unmatched_det.append(c)
            else:
                matched.append((r, c))

        # 更新匹配轨迹
        for trk_idx, det_idx in matched:
            track = self.tracks[trk_idx]
            measurement = self._convert_bbox_to_z(detections[det_idx])
            track.mean, track.cov = self.tracker.update(track.mean, track.cov, measurement)
            track.time_since_update = 0
            track.age += 1

        # 创建新轨迹
        for det_idx in unmatched_det:
            meas = self._convert_bbox_to_z(detections[det_idx])
            mean, cov = self.tracker.initiate(meas)
            new_track = Track(mean, cov, self.track_id)
            self.track_id += 1
            self.tracks.append(new_track)

        # 删除老轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        # 返回活跃轨迹
        outputs = []
        for t in self.tracks:
            if t.age >= self.min_hits:
                bbox = t.to_bbox()
                outputs.append([*bbox, t.track_id])
        return outputs

    def _convert_bbox_to_z(self, bbox):
        x1, y1, x2, y2 = bbox[:4]
        # 防止宽高为负或0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w/2
        cy = y1 + h/2
        s = w*h
        r = w/h
        return np.array([cx, cy, s, r])
