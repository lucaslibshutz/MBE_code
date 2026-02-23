import numpy as np

def calculate_ellipse(X, P, nsig=1, np_points=36):
    """
    Calculate points of a 2D covariance ellipse.

    Parameters
    ----------
    X : array-like, shape (2,)
        2D mean vector
    P : array-like, shape (2,2)
        2D covariance matrix
    nsig : float, optional
        Number of sigmas (default = 1)
    np_points : int, optional
        Number of ellipse points (default = 36)

    Returns
    -------
    Xe : ndarray
        Ellipse x-coordinates
    Ye : ndarray
        Ellipse y-coordinates
    U : ndarray
        Left singular vectors (rotation matrix)
    S : ndarray
        Singular value matrix
    th : float
        Rotation angle in degrees

    MAE 6760 Model Based Estimation
    Cornell University
    M Campbell
    """

    X = np.asarray(X).reshape(-1)
    P = np.asarray(P)

    if X.size != 2 or P.shape != (2, 2):
        raise ValueError("Mean and covariance must be 2D")

    # SVD of covariance
    U, S_vals, Vt = np.linalg.svd(P)
    S = np.diag(S_vals)

    # Make U a true rotation matrix (match MATLAB sign correction)
    if U[0, 0] != 0:
        U = U @ np.diag(np.sign(np.diag(U)))

    # Rotation angle (degrees)
    th = -np.arcsin(U[0, 1]) * 180.0 / np.pi

    # Semi-axis lengths
    s1 = nsig * np.sqrt(S[0, 0])
    s2 = nsig * np.sqrt(S[1, 1])

    x, y = X

    # Angle definitions
    beta = np.deg2rad(th)
    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)

    alpha = np.linspace(0, 360, np_points) * np.pi / 180.0
    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    # Ellipse parametric equations
    Xe = x + (s1 * cosalpha * cosbeta - s2 * sinalpha * sinbeta)
    Ye = y + (s1 * cosalpha * sinbeta + s2 * sinalpha * cosbeta)

    return Xe, Ye, U, S, th