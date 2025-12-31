import numpy as np

class Detection:
    def __init__(self, tlbr, confidence, feature):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_xyah(self):
        x1, y1, x2, y2 = self.tlbr
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.
        cy = y1 + h / 2.
        return np.array([cx, cy, w / h, h])
