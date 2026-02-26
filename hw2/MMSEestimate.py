import numpy as np

def mmse_estimate(z, R, x0, Px0, H=None):
    """
    Calculates the MMSE estimate and error covariance
    with a prior and n datapoints.

    Measurement model:
        z = H x + v

    Parameters
    ----------
    z : ndarray (nz, n)
        Measurement matrix
    R : ndarray (nz, nz) or scalar
        Measurement noise covariance
    x0 : ndarray (nx,) or scalar
        Prior mean
    Px0 : ndarray (nx, nx) or scalar
        Prior covariance
    H : ndarray (nz, nx) or scalar, optional
        Measurement matrix (default: scalar case H=1)

    Returns
    -------
    xhat : ndarray
        MMSE estimate
    Pxhat : ndarray
        MMSE error covariance

    MAE 6760 Model Based Estimation
    Cornell University
    M Campbell    
    """

    if H is None:
        H = 1.0

    nz, n = z.shape
    nx = x0.size

    H = np.atleast_2d(H)
    if H.shape == (1, 1):
        H = H * np.ones((nz, nx))

    # Initialize
    M1 = np.linalg.inv(Px0)
    M2 = M1 @ x0

    Rinv = np.linalg.inv(R)

    # MMSE summation
    for k in range(n):
        M1 = M1 + H.T @ Rinv @ H
        M2 = M2 + H.T @ Rinv @ z[:, k]

    Pxhat = np.linalg.inv(M1)
    xhat = Pxhat @ M2

    return xhat.squeeze(), Pxhat.squeeze()

def mmse_estimate_batch(z, H, R, x0, P0):
    """
    Batch MMSE / Bayesian Linear Regression

    """
    z = np.asarray(z).squeeze()
    H = np.asarray(H)
    x0 = np.asarray(x0).squeeze()
    P0 = np.asarray(P0)

    n, nx = H.shape
    assert z.shape == (n,), f"Expected z shape (n,), got {z.shape}"
    assert x0.shape == (nx,), f"Expected x0 shape (nx,), got {x0.shape}"
    assert P0.shape == (nx, nx), f"Expected P0 shape (nx, nx), got {P0.shape}"

    # Build R
    R = np.asarray(R)
    if R.ndim == 1: # only variances
        W = np.diag(1.0 / R**2)
    else:
        # R is covariance matrix
        W = np.linalg.inv(R) 

    P0_inv = np.linalg.inv(P0)
    S = P0_inv + H.T @ W @ H
    b = P0_inv @ x0 + H.T @ W @ z
    Pxhat = np.linalg.inv(S)
    xhat = Pxhat @ b
    xhat.reshape(-1,1)
    return xhat, Pxhat
