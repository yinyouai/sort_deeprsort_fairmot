# sort/kalman_filter.py
import numpy as np

class KalmanFilter:
    def __init__(self):
        # 8D state: [x, y, s, r, vx, vy, vs, vr]
        ndim, dt = 4, 1.

        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim)

        self._std_position = 1. / 20
        self._std_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        covariance = np.eye(8) * 10.
        return mean, covariance

    def predict(self, mean, covariance):
        mean = np.dot(self._motion_mat, mean)
        covariance = np.dot(self._motion_mat, np.dot(covariance, self._motion_mat.T))
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean = mean[:4]
        projected_cov = covariance[:4, :4]

        R = np.eye(4)
        S = projected_cov + R
        K = np.dot(covariance[:, :4], np.linalg.inv(S))

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(K, innovation)
        new_cov = covariance - np.dot(K, projected_cov).dot(K.T)
        return new_mean, new_cov
