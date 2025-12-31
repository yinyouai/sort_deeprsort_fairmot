import numpy as np


def iou(bbox, candidates):
    bbox_tl = bbox[:2]
    bbox_br = bbox[2:]

    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)

    wh = np.maximum(0., br - tl)
    area_intersection = wh[:, 0] * wh[:, 1]

    area_bbox = np.prod(bbox_br - bbox_tl)
    area_candidates = np.prod(candidates_br - candidates_tl, axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices, detection_indices):
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        bbox = tracks[track_idx].to_tlbr()
        candidates = np.asarray([
            detections[i].tlbr for i in detection_indices
        ])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
