import numpy as np

def beaconNLS(z, R, x0, xS):
    """
    z = h(x) + v
    where z = [RA, RB, RC], and each is a distance to a becon.

    Parameters
    ----------
    z: ndarray (3,n)
    Measurement matrix [RA, RB, RC]
    R: ndarray (3,3)
    Measurement noise covariance
    x0: ndarray (2,)
    Initial parameter estimate
    xS : ndarray (3,2)
    Sensor locations RA, RB, RC

    Returns
    -------
    xhatF: ndarray (2,)
    Final position estimate
    PxhatF: ndarray (2,2)
    Final covariance estimate
    Hhat: ndarray (2,3)
    Final linearized H matrix
    iter_count: int
    Number of iterations
    """
    tol = 1e-3
    max_iter = 100

    z = np.asarray(z)
    xhat = x0.reshape(2,1)

    nz, n = z.shape
    Rinv = np.linalg.inv(R)

    assert np.shape(xS) == (3,2), f"xS is incorrect shape of {np.shape(xS)}"

    J_old = 1.0
    iter_count = 0

    for _ in range(max_iter):

        iter_count += 1
        xi = xhat[:,-1].reshape(-1,1)

        dxA = xi[0,0] - xS[0,0]
        dyA = xi[1,0] - xS[0,1]

        dxB = xi[0,0] - xS[1,0]
        dyB = xi[1,0] - xS[1,1]

        dxC = xi[0,0] - xS[2,0]
        dyC = xi[1,0] - xS[2,1]

        rhatA = np.sqrt(dxA**2 + dyA**2)
        rhatB = np.sqrt(dxB**2 + dyB**2)
        rhatC = np.sqrt(dxC**2 + dyC**2)

        zhat = np.array([rhatA, rhatB, rhatC]).reshape(-1,1)

        # Jacobian
        Hhat = np.array([
            [dxA/rhatA, dyA/rhatA],
            [dxB/rhatB, dyB/rhatB],
            [dxC/rhatC, dyC/rhatC],
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
    xhatF = xhat[:, -1]

    return xhatF, PxhatF, Hhat, iter_count

