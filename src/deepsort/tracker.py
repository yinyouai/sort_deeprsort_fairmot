import numpy as np

from .kalman_filter import KalmanFilter
from .linear_assignment import min_cost_matching
from .iou_matching import iou_cost
from .track import Track, TrackState


class Tracker:
    def __init__(
        self,
        metric,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3
    ):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # === 1. appearance matching ===
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # === 2. update matched tracks ===
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx]
            )

        # === 3. mark unmatched tracks ===
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # === 4. create new tracks ===
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # === 5. remove deleted tracks ===
        self.tracks = [
            t for t in self.tracks
            if t.state != TrackState.Deleted
        ]

        # === 6. update appearance metric ===
        active_tracks = [
            t.track_id for t in self.tracks
            if t.is_confirmed()
        ]
        features, targets = [], []

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id] * len(track.features)
            track.features = []

        self.metric.partial_fit(
            np.asarray(features),
            np.asarray(targets),
            active_tracks
        )

    def _match(self, detections):
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks)
            if t.is_confirmed()
        ]

        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks)
            if not t.is_confirmed()
        ]

        # === appearance matching for confirmed tracks ===
        matches_a, unmatched_tracks_a, unmatched_detections = \
            min_cost_matching(
                self._appearance_cost,
                self.metric.matching_threshold,
                self.tracks,
                detections,
                confirmed_tracks,
                list(range(len(detections)))
            )

        # === IoU matching for unconfirmed tracks ===
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(
                iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                unconfirmed_tracks,
                unmatched_detections
            )

        matches = matches_a + matches_b
        unmatched_tracks = unmatched_tracks_a + unmatched_tracks_b

        return matches, unmatched_tracks, unmatched_detections

    def _appearance_cost(
        self,
        tracks,
        detections,
        track_indices,
        detection_indices
    ):
        features = np.array([
            detections[i].feature for i in detection_indices
        ])
        targets = [
            tracks[i].track_id for i in track_indices
        ]
        return self.metric.distance(features, targets)

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                detection.feature
            )
        )
        self._next_id += 1
