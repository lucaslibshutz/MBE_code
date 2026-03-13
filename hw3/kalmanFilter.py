import numpy as np
from numpy.typing import ArrayLike

def kalman_filter(
    F: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    Q: ArrayLike,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    z: np.ndarray,
    nk: int,
    Kbar: np.ndarray = None,
    Lambda0: ArrayLike = None
):
    nx = np.shape(x0)[0]
    xhat_P = np.zeros((nx, nk))
    xhat_U = np.zeros((nx, nk))
    xhat_P[:, 0] = x0
    xhat_U[:, 0] = x0

    Pxhat_P = np.zeros((nx, nx, nk))
    Pxhat_U = np.zeros((nx, nx, nk))
    Pxhat_P[:, :, 0] = P0
    Pxhat_U[:, :, 0] = P0
    Q = np.atleast_2d(Q)

    for k in range(nk-1):
        xhat_P[:,k+1] = F @ xhat_U[:,k]
        Pxhat_P[:,:,k+1] = F @ Pxhat_U[:,:,k] @ F.T + G @ Q @ G.T

        xPred = xhat_P[:,k+1]
        PxPred = Pxhat_P[:,:,k+1]

        # Kalman Gain
        K_ins = H @ PxPred @ H.T + R # both masses
        if Kbar is not None:
            K = Kbar
        else:
            K = (PxPred @ H.T)@ np.linalg.inv(K_ins) # both masses

        # Update step
        if z.ndim == 1:
            if Lambda0 is not None:
                inn = z[k+1] - H @ xPred
                S = H @ PxPred @ H.T + R
                Lam = inn.T @ np.linalg.inv(S) @ inn
                if Lam > Lambda0:
                    xhat_U[:,k+1] = xhat_P[:,k+1]
                    Pxhat_U[:,:,k+1] = Pxhat_P[:,:,k+1]
                else:
                    xhat_U[:,k+1] = xPred + K @ inn # both masses
                    origTerm = np.eye(nx) - K @ H # both masses
                    Pxhat_U[:, :, k+1] = origTerm @ PxPred @ origTerm.T + K @ R @ K.T # both masses
            else:
                xhat_U[:,k+1] = xPred + K @ (z[k+1]- H @ xPred) # both masses
                origTerm = np.eye(nx) - K @ H # both masses
                Pxhat_U[:, :, k+1] = origTerm @ PxPred @ origTerm.T + K @ R @ K.T # both masses

        else:
            if Lambda0 is not None:
                inn = z[:,k+1] - H @ xPred
                S = H @ PxPred @ H.T + R
                Lam = inn.T @ np.linalg.inv(S) @ inn
                if Lam > Lambda0:
                    xhat_U[:,k+1] = xhat_P[:,k+1]
                    Pxhat_U[:,:,k+1] = Pxhat_P[:,:,k+1]
                else:
                    xhat_U[:,k+1] = xPred + K @ inn # both masses
                    origTerm = np.eye(nx) - K @ H # both masses
                    Pxhat_U[:, :, k+1] = origTerm @ PxPred @ origTerm.T + K @ R @ K.T # both masses
            else:
                xhat_U[:,k+1] = xPred + K @ (z[:,k+1]- H @ xPred) # both masses
                origTerm = np.eye(nx) - K @ H # both masses
                Pxhat_U[:, :, k+1] = origTerm @ PxPred @ origTerm.T + K @ R @ K.T # both masses
    return xhat_P, xhat_U, Pxhat_P, Pxhat_U
