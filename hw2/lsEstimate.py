import numpy as np

def LSestimate(z, H, R=None):
    """
    Least squares parameter estimate.

    Measurement model: z = H x + v
    
    Args:
        z (np.ndarray (nz,n)): Measurement data
        H (np.ndarray (nz,nx)): Measurement matrix
        R (np.ndarray (nz, nz), optional): Measurement noise covaraince (default = I)

    Returns:
        xhat (np.ndarray (nx,)): Least Squares Estimate
        Pxhat (np.ndarray (nx,nx)): Estimate covariance
    """
    nz, n = z.shape
    _, nx = H.shape
    
    if R is None:
        R = np.eye(nz)

    Rinv = np.linalg.inv(R)

    M1 = np.zeros((nx,nx))
    M2 = np.zeros((nx,1))

    for k in range(n):
        zk = z[:,[k]]
        M1 += H.T @ Rinv @ H
        M2 += H.T @ Rinv @ zk

    Pxhat = np.linalg.inv(M1)
    xhat = Pxhat @ M2

    return xhat.squeeze(), Pxhat.squeeze()
