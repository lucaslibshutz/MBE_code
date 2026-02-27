import numpy as np

def beaconNLS(z, R, x0, xS,xTrue=None):
    """
    z = h(x) + v
    where z = [RA, RB, RC], and each is a distance to a becon.

    Parameters
    ----------
    z: ndarray (m,n)
    Measurement matrix [RA, RB, RC]
    R: ndarray (m,m)
    Measurement noise covariance
    x0: ndarray (2,)
    Initial parameter estimate
    xS : ndarray (m,2)
    Beacon locations, each row is [x_i, y_i]
    xTrue: ndarray (2,), optional
    True position, used for perfect linearization of Jacobian if included 

    Returns
    -------
    xhatF: ndarray (2,)
    Final position estimate
    PxhatF: ndarray (2,2)
    Final covariance estimate
    Hhat: ndarray (m,2)
    Final linearized H matrix
    iter_count: int
    Number of iterations
    """
    tol = 1e-3
    max_iter = 100

    z = np.asarray(z)
    xhat = x0.reshape(2,1)

    nz, n = z.shape
    m = nz

    assert xS.shape == (m,2), f"xS shape {xS.shape} must be (m,2 ) where m={m}"
    assert R.shape == (m,m), f"R shape {R.shape} must be (m,m) where m={m}"


    Rinv = np.linalg.inv(R)

    J_old = 1.0
    iter_count = 0

    if xTrue is not None:
        x_lin = np.asarray(xTrue).reshape(2,)
        dx = x_lin[0] - xS[:,0] # shape (m,)
        dy = x_lin[1] - xS[:,1] # shape (m,)
        rhat = np.sqrt(dx**2 + dy**2) # shape (m,)
        
        zhat = rhat.reshape(m,1)
        Hhat = np.column_stack((dx/rhat, dy/rhat)) # shape (m,2)

        # Precompute Rinv and the constant normal matrix
        M1_const = n * (Hhat.T @ Rinv @ Hhat)
        
        sum_z = np.sum(z,axis=1).reshape(m,1)
        sum_e = sum_z - n * zhat
        M2_const = Hhat.T @ Rinv @ sum_e

        # Solve once for batched constant H matrix
        dxhat = np.linalg.inv(M1_const) @ M2_const
        xhatF = x_lin + dxhat.flatten()
        PxhatF = np.linalg.inv(M1_const)

        return xhatF, PxhatF, Hhat

    else:
        for _ in range(max_iter):

            iter_count += 1
            xi = xhat[:,-1].reshape(-1,1)

            # Vectorize dx/dy/rhat
            dx = xi[0,0] - xS[:,0] # shape (m,)
            dy = xi[1,0] - xS[:,1] # shape (m,)
            rhat = np.sqrt(dx**2 + dy**2) # shape (m,)
            
            zhat = rhat.reshape(m,1)

            # Jacobian
            Hhat = np.column_stack((dx/rhat, dy/rhat)) # shape (m,2)

            M1 = np.zeros((2,2))
            M2 = np.zeros((2,1))

            J = 0.0

            for k in range(n):
                ek = z[:, k:k+1] - zhat
                M1 += Hhat.T @ Rinv @ Hhat
                M2 += Hhat.T @ Rinv @ ek
                J += 0.5 * ek.T @ Rinv @ ek


            # Try except in case solution is ill-conditioned
            try:
                dxhat = np.linalg.inv(M1) @ M2
            except np.linalg.LinAlgError:
                dxhat = np.linalg.pinv(M1) @ M2

            x_next = xi + dxhat
            xhat = np.hstack((xhat, x_next))

            if abs(J_old - J.item()) < tol:
                break
            J_old = J.item()

        PxhatF = np.linalg.inv(M1)
        xhatF = xhat[:, -1]

        return xhatF, PxhatF, Hhat, iter_count
