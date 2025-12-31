import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between two matrices."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2 = np.sum(a ** 2, axis=1)
    b2 = np.sum(b ** 2, axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    return np.clip(r2, 0., float(np.inf))


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


class NearestNeighborDistanceMetric:
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _pdist
        elif metric == "cosine":
            self._metric = _cosine_distance
        else:
            raise ValueError("Invalid metric; must be 'euclidean' or 'cosine'")

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

        self.samples = {
            k: self.samples[k]
            for k in active_targets
            if k in self.samples
        }

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = np.min(
                self._metric(self.samples[target], features), axis=0
            )
        return cost_matrix
