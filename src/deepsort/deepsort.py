import numpy as np

from .tracker import Tracker
from .detection import Detection
from .nn_matching import NearestNeighborDistanceMetric
from .reid.extractor import Extractor


class DeepSort:
    def __init__(
        self,
        model_path,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100
    ):
        """
        DeepSORT tracker wrapper

        Args:
            model_path (str): ReID model checkpoint
            max_dist (float): cosine distance threshold
            max_iou_distance (float): IoU threshold for secondary matching
            max_age (int): max missed frames before track deletion
            n_init (int): frames required to confirm a track
            nn_budget (int): max features stored per track
        """

        # ReID feature extractor
        self.extractor = Extractor(model_path)

        # Appearance distance metric (cosine)
        metric = NearestNeighborDistanceMetric(
            metric="cosine",
            matching_threshold=max_dist,
            budget=nn_budget
        )

        # Multi-object tracker
        self.tracker = Tracker(
            metric=metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )

    def update(self, boxes, scores, image=None):
        """
        Update tracker with detections of one frame

        Args:
            boxes (ndarray): Nx4, [x1, y1, x2, y2]
            scores (ndarray): Nx1, detection confidence
            image (ndarray | None): BGR image for ReID (optional)

        Returns:
            list: [track_id, x1, y1, x2, y2]
        """

        if len(boxes) == 0:
            self.tracker.predict()
            self.tracker.update([])
            return []

        # === 1. extract appearance features ===
        # 从图片中提取每个检测框对应的区域
        imgs = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                imgs.append(crop)
            else:
                # 如果框无效，使用整个图像
                imgs.append(image)
        features = self.extractor(imgs)

        # === 2. build Detection objects ===
        detections = []
        for bbox, score, feat in zip(boxes, scores, features):
            detections.append(
                Detection(bbox, score, feat)
            )

        # === 3. run tracker ===
        self.tracker.predict()
        self.tracker.update(detections)

        # === 4. collect confirmed tracks ===
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update > 0:
                continue

            x1, y1, x2, y2 = track.to_tlbr()
            outputs.append([
                track.track_id,
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ])

        return outputs
