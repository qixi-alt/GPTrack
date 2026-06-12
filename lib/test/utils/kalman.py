import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Kf_Params:
    B = 0  # External input is zero.
    u = 0  # External input is zero.
    K = float('nan')  # Kalman gain does not need initialization.
    z = float('nan')  # No initialization is needed here; set observation z before each kf_update call.
    P = np.diag(np.ones(4))  # Initial P; originally considered zeros(4, 4).

    # Initial state: provided externally and initialized from observations; vx and vy start at 0.
    x = []
    G = []

    # State transition matrix A.
    # This encodes a linear motion model where previous position plus velocity gives current position, and velocity stays constant.
    A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)

    # Prediction noise covariance Q assumes Gaussian noise in the prediction process.
    # Its scale reflects trust in the prediction process. Increase relevant diagonal entries for less stable motion.
    # Lower values can produce smoother trajectories.
    Q = np.diag(np.ones(4)) * 0.1

    # Observation matrix H: z = H * x.
    # The state is (x, y, vx, vy) and observations are (x, y), so H = eye(2, 4).
    H = np.eye(2, 4)

    # Observation noise covariance R assumes Gaussian noise in the observation process.
    # Its scale reflects trust in observations; a reliable x coordinate should use a smaller first value.
    R = np.diag(np.ones(2)) * 0.1


def kf_init(px, py, vx, vy):
    # Here state x is (x, y, vx, vy) and observation z is (x, y).
    kf_params = Kf_Params()
    kf_params.B = 0
    kf_params.u = 0
    kf_params.K = float('nan')
    kf_params.z = float('nan')
    kf_params.P = np.diag(np.ones(4))
    kf_params.x = [px, py, vx, vy]
    kf_params.G = [px, py, vx, vy]
    kf_params.A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)
    kf_params.Q = np.diag(np.ones(4)) * 0.1
    kf_params.H = np.eye(2, 4)
    kf_params.R = np.diag(np.ones(2)) * 0.1
    return kf_params


def kf_update(kf_params):
    # The following are the five Kalman-filter equations.
    a1 = np.dot(kf_params.A, kf_params.x)
    a2 = kf_params.B * kf_params.u
    x_ = np.array(a1) + np.array(a2)

    b1 = np.dot(kf_params.A, kf_params.P)
    b2 = np.dot(b1, np.transpose(kf_params.A))
    p_ = np.array(b2) + np.array(kf_params.Q)

    c1 = np.dot(p_, np.transpose(kf_params.H))
    c2 = np.dot(kf_params.H, p_)
    c3 = np.dot(c2, np.transpose(kf_params.H))
    c4 = np.array(c3) + np.array(kf_params.R)
    c5 = np.linalg.matrix_power(c4, -1)
    kf_params.K = np.dot(c1, c5)

    d1 = np.dot(kf_params.H, x_)
    d2 = np.array(kf_params.z) - np.array(d1)
    d3 = np.dot(kf_params.K, d2)
    kf_params.x = np.array(x_) + np.array(d3)

    e1 = np.dot(kf_params.K, kf_params.H)
    e2 = np.dot(e1, p_)
    kf_params.P = np.array(p_) - np.array(e2)

    kf_params.G = x_
    return kf_params


def accuracy(predictions, labels):
    return np.array(predictions) - np.array(labels)
