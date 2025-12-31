import numpy as np
from scipy.optimize import linear_sum_assignment


def min_cost_matching(
    distance_metric,
    max_distance,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None
):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_indices[row])
            unmatched_detections.append(detection_indices[col])
        else:
            matches.append((track_indices[row], detection_indices[col]))

    return matches, unmatched_tracks, unmatched_detections
