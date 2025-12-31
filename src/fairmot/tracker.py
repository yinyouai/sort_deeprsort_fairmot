import numpy as np
from scipy.optimize import linear_sum_assignment

class Detection:
    def __init__(self, x1, y1, x2, y2, score):
        self.bbox = [x1, y1, x2, y2]
        self.score = score

class Track:
    def __init__(self, track_id, bbox, embedding):
        self.track_id = track_id
        self.bbox = bbox
        self.embedding = embedding
        self.time_since_update = 0

class Tracker:
    def __init__(self, max_age=30):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age

    def update(self, detections, embeddings):
        if len(self.tracks) == 0:
            for det, emb in zip(detections, embeddings):
                self.tracks.append(Track(self.next_id, det.bbox, emb))
                self.next_id += 1
            return [(t.track_id, t.bbox) for t in self.tracks]

        # cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou = self.iou(track.bbox, det.bbox)
                emb_dist = np.linalg.norm(track.embedding - embeddings[j])
                cost_matrix[i, j] = 0.5 * (1 - iou) + 0.5 * emb_dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks, assigned_dets = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.7:
                self.tracks[r].bbox = detections[c].bbox
                self.tracks[r].embedding = embeddings[c]
                self.tracks[r].time_since_update = 0
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # unmatched detections -> new tracks
        for idx, det in enumerate(detections):
            if idx not in assigned_dets:
                self.tracks.append(Track(self.next_id, det.bbox, embeddings[idx]))
                self.next_id += 1

        # remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [(t.track_id, t.bbox) for t in self.tracks]

    @staticmethod
    def iou(bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
