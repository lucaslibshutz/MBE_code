import numpy as np

def nonlinearLS(z, R, x0, xS):
    """
    Nonlinear Least Squares for range & bearing measurements.
    Measurement model:
    z = h(x) + v
    where
    z = [range; bearing]
    Parameters
    ----------
    z : ndarray (2, n)
    Measurement matrix [range; bearing]
    R : ndarray (2,2)
    Measurement noise covariance
    x0 : ndarray (2,)
    Initial parameter estimate
    xS : ndarray (2,)
    Sensor location
    Returns
    -------
    xhatF : ndarray (2,)
    Final parameter estimate
    PxhatF : ndarray (2,2)
    Final covariance estimate
    Hhat : ndarray (2,2)
    Final linearized H matrix
    iter_count : int
    Number of iterations
    MAE 6760 Model Based Estimation
    Cornell University
    M Campbell
    """
    tol = 1e-3
    max_iter = 100

    z = np.asarray(z)
    xhat = x0.reshape(2,1)

    nz, n = z.shape
    Rinv = np.linalg.inv(R)

    J_old = 1.0
    iter_count = 0

    for _ in range(max_iter):

        iter_count += 1
        xi = xhat[:, -1].reshape(2,1)

        dx = xi[0,0] - xS[0]
        dy = xi[1,0] - xS[1]

        Rhat = np.sqrt(dx**2 + dy**2)
        Thhat = np.arctan2(dy,dx)

        zhat = np.array([Rhat, Thhat]).reshape(-1,1)

        # Jacobian (H matrix)
        Hhat = np.array([
            [dx / Rhat, dy / Rhat],
            [-dy / (Rhat ** 2), dx / (Rhat ** 2)]
        ])

        M1 = np.zeros((2,2))
        M2 = np.zeros((2,1))
        J = 0.0

        for k in range(n):
            ek = z[:, k:k+1] - zhat
            M1 += Hhat.T @ Rinv @ Hhat
            M2 += Hhat.T @ Rinv @ ek
            J += 0.5 * ek.T @ Rinv @ ek

        dxhat = np.linalg.inv(M1) @ M2
        x_next = xi + dxhat

        xhat = np.hstack((xhat, x_next))

        if abs(J_old - J.item()) < tol:
            break
        J_old = J.item()
        
    PxhatF = np.linalg.inv(M1)
    xhatF = xhat[:,-1]

    return xhatF, PxhatF, Hhat, iter_count


