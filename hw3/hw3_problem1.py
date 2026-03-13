import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import addcopyfighandler #noqa F401
from kalmanFilter import kalman_filter

from plot_openloop import plot_openloop
from plot_estimator import plot_estimator

# environment vars
plot_I = 0
plot_A = 0
plot_D = 1

# Define CT system
k1 = k2 = k3 = k4 = k5 = 1
c1 = c2 = c3 = c4 = c5 = 0.01
m1 = m2 = m3 = m4 = m5 = 1

nx = 10
Fc = np.zeros((nx,nx))

# stiffness
Fc[np.ix_([1,3,5,7,9],[0,2,4,6,8])] = np.array([
[-(k1+k2)/m1, (k2)/m1, 0, 0, 0],
[ (k2)/m2, -(k2+k3)/m2, (k3)/m2, 0, 0],
[ 0, (k3)/m3, -(k3+k4)/m3, (k4)/m3, 0],
[ 0, 0, (k4)/m4, -(k4+k5)/m4, (k5)/m4],
[ 0, 0, 0, (k5)/m5, -(k5)/m5]
])
# damping
Fc[np.ix_([1,3,5,7,9],[1,3,5,7,9])] = np.array([
[-(c1+c2)/m1, (c2)/m1, 0, 0, 0],
[ (c2)/m2, -(c2+c3)/m2, (c3)/m2, 0, 0],
[0,(c3)/m3, -(c3+c4)/m3,(c4)/m3,0],
[0,0,(c4)/m4, -(c4+c5)/m4,(c5)/m4],
[0,0,0,(c5)/m5, -(c5)/m5]
])
# kinematics
Fc[np.ix_([0,2,4,6,8],[1,3,5,7,9])] = np.eye(5)

# disturbance on fifth mass
Gc = np.zeros((nx,1))
Gc[9,0] = 1/m5

# position measurements on 1st and 5th measurements (essentially a selector matrix)
i1 = 0
Hq1 = np.zeros((1,nx))
Hq1[0,i1] = 1
i5 = 8
Hq5 = np.zeros((1,nx))
Hq5[0,i5] = 1
Hc = np.vstack((Hq1,Hq5))

# discrete time model
dt = 0.05
F, G, _, _, _ = cont2discrete((Fc, Gc, Hc, 0), dt)

# Simulate system
t = np.arange(0,50+dt,dt)
nk = len(t)

x0 = np.zeros(nx)
x0[8] = 1

np.random.seed(101)
Qsim = 0.01 # process noise covariance
w = np.sqrt(Qsim) * np.random.randn(1,nk)

x_true = np.zeros((nx,nk))
x_no_w = np.zeros((nx,nk))
x_true[:,0] = x0
x_no_w[:,0] = x0

for k in range(nk-1):
    x_no_w[:, k+1] = F @ x_no_w[:,k]
    x_true[:, k+1] = F @ x_true[:,k] + G @ w[:,k]

# measurements
Rq1 = 0.001
Rq5 = 0.001
z_q1 = (Hq1 @ x_true + np.sqrt(Rq1) * np.random.randn(1,nk)).ravel() # scalar measurement
z_q5 = (Hq5 @ x_true + np.sqrt(Rq5) * np.random.randn(1,nk)).ravel() # scalar measurement

# open loop plots
if plot_I:
    fig, axs = plt.subplots(1,2, figsize=(16,6))

    plot_openloop(t, x_true[i1, :], z=z_q1, x_no_w=x_no_w[i1,:],ax=axs[0])
    axs[0].set_ylabel("position of first mass")

    plot_openloop(t, x_true[i5, :], z=z_q5, x_no_w=x_no_w[i5,:],ax=axs[1])
    axs[1].set_ylabel("position of fifth mass")

    fig.suptitle("Siimulated Response: 1st and 5th Masses",fontweight='bold')

    plt.tight_layout()
    plt.show()

### --------------------------------------
### (a) KF using z_q1
# KF for each mass, and see the uncertainty in the other
xhat_pA1, xhat_uA1, Pxhat_pA1, Pxhat_uA1 = kalman_filter(
    F=F,
    G=G,
    H=Hq1,
    Q=Qsim,
    R=np.diag([Rq1]),
    z=z_q1,
    x0=x0,
    P0=1/4*np.eye(nx),
    nk=nk
)

xhat_pA5, xhat_uA5, Pxhat_pA5, Pxhat_uA5 = kalman_filter(
    F=F,
    G=G,
    H=Hq5,
    Q=Qsim,
    R=np.diag([Rq5]),
    z=z_q5,
    x0=x0,
    P0=1/4*np.eye(nx),
    nk=nk
)

if plot_A:
    # Plot results

    ## -- First mass KF --
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    # First mass
    plot_estimator(t, xhat_uA1[i1,:], Pxhat_uA1[i1,i1,:],x_true[i1,:],plot_type='state',z=z_q1,ax=axs[0])
    axs[0].set_ylabel(r"position of first mass $m_1$")

    # Fifth mass
    plot_estimator(t, xhat_uA1[i5,:], Pxhat_uA1[i5,i5,:],x_true[i5,:],plot_type='state',z=None,ax=axs[1])
    axs[1].set_ylabel(r"position of fifth mass $m_5$")
    fig.suptitle("First mass KF",fontweight='bold')
    plt.tight_layout()
    plt.show()

    ## -- Fifth mass KF --
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    # First mass
    plot_estimator(t, xhat_uA5[i1,:], Pxhat_uA5[i1,i1,:],x_true[i1,:],plot_type='state',z=None,ax=axs[0])
    axs[0].set_ylabel(r"position of first mass $m_1$")

    # Fifth mass
    plot_estimator(t, xhat_uA5[i5,:], Pxhat_uA5[i5,i5,:],x_true[i5,:],plot_type='state',z=z_q5,ax=axs[1])
    axs[1].set_ylabel(r"position of fifth mass $m_5$")
    fig.suptitle("Fifth mass KF",fontweight='bold')
    plt.tight_layout()
    plt.show()



    # -- Mass 1 KF -- 
    fig,axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pA1[i1,:], Pxhat_pA1[i1,i1,:],x_true[i1,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_uA1[i1,:], Pxhat_uA1[i1,i1,:],x_true[i1,:],plot_type='error',z=z_q1,ax=axs[1])
    axs[0].set_title(r"Predicted Error $(q_1)$")
    axs[1].set_title(r"Updated Error $(q_1)$")
    fig.suptitle("Mass 1 KF")
    fig.tight_layout()
    plt.show()

    fig,axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pA1[i5,:], Pxhat_pA1[i5,i5,:],x_true[i5,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_uA1[i5,:], Pxhat_uA1[i5,i5,:],x_true[i5,:],plot_type='error',z=None,ax=axs[1])
    axs[0].set_title(r"Predicted Error $(q_5)$")
    axs[1].set_title(r"Updated Error $(q_5)$")
    fig.suptitle("Mass 1 KF")
    fig.tight_layout()
    plt.show()

    # -- Mass 5 KF --
    fig,axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pA5[i1,:], Pxhat_pA5[i1,i1,:],x_true[i1,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_uA5[i1,:], Pxhat_uA5[i1,i1,:],x_true[i1,:],plot_type='error',z=None,ax=axs[1])
    axs[0].set_title(r"Predicted Error $(q_1)$")
    axs[1].set_title(r"Updated Error $(q_1)$")
    fig.suptitle("Mass 5 KF")
    fig.tight_layout()
    plt.show()

    fig,axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pA5[i5,:], Pxhat_pA5[i5,i5,:],x_true[i5,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_uA5[i5,:], Pxhat_uA5[i5,i5,:],x_true[i5,:],plot_type='error',z=z_q5,ax=axs[1])
    axs[0].set_title(r"Predicted Error $(q_5)$")
    axs[1].set_title(r"Updated Error $(q_5)$")
    fig.suptitle("Mass 5 KF")
    fig.tight_layout()
    plt.show()

    ### --------------------------------------
    ### (b) Predicted vs Updated Covariances for mass 1 & 5
    ### --------------------------------------

    # Plot variances and error of fifth mass for predicted and updated
    fig, axs = plt.subplots(1,2, figsize=(16,6))

    axs[0].semilogy(t, Pxhat_pA1[i1,i1,:], 'r',label='Predicted')
    axs[0].semilogy(t, Pxhat_uA1[i1,i1,:], 'b',label='Updated')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel(r"$\sigma^2(m_1)$")
    axs[0].legend()

    axs[1].semilogy(t, Pxhat_pA1[i5,i5,:],'r', label='Predicted')
    axs[1].semilogy(t, Pxhat_uA1[i5,i5,:],'b', label='Updated')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel(r"$\sigma^2(m_5)$")
    axs[1].legend()
    fig.suptitle("Mass 1 KF: Predicted vs Updated Variance",fontweight='bold')
    fig.tight_layout()
    plt.show()

### --------------------------------------
### (c) Steady State Error Covariance
### --------------------------------------
Pbar1 = solve_discrete_are(a=F.T, b=Hq1.T, q=G @ np.atleast_2d(Qsim) @ G.T, r=np.diag([Rq1]))
Pbar5 = solve_discrete_are(a=F.T, b=Hq5.T, q=G @ np.atleast_2d(Qsim) @ G.T, r=np.diag([Rq5]))

print(f"Simulated measurement covariance for mass 1 at the final time is: {(Pxhat_uA1[i1,i1,nk-1]).astype(float)}")
print(f"Steady state error covariance for mass 1 is: {(Hq1 @ Pbar1 @ Hq1.T).astype(float)}")

print(f"Simulated measurement covariance for mass 5 at the final time is: {(Pxhat_uA5[i5,i5,nk-1]).astype(float)}")
print(f"Steady state error covariance for mass 5 is: {(Hq5 @ Pbar5 @ Hq5.T).astype(float)}")


### --------------------------------------
### (d) Constant gain KF
### --------------------------------------
# Perform the same KF from (a), but this time use P0 = Pbar, and constant Kalman gain Kbar = Pbar @ H.T @ inv(H @ Pbar @ H.T + R)
# Initialize variables
Kbar = Pbar5 @ Hq5.T @ np.linalg.inv(Hq5 @ Pbar5 @ Hq5.T + np.diag([Rq5]))
xhat_pD, xhat_uD, Pxhat_pD, Pxhat_uD = kalman_filter(
    F=F,
    G=G,
    H=Hq5,
    Q=Qsim,
    R=np.diag([Rq5]),
    z=z_q5,
    x0=x0,
    P0=Pbar5,
    nk=nk,
    Kbar=Kbar
)


# Plot estimator for part (D)
if plot_D:
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_uD[i1,:], Pxhat_uD[i1,i1,:],x_true[i1,:],plot_type='state',z=None,ax=axs[0])
    plot_estimator(t, xhat_uD[i5,:], Pxhat_uD[i5,i5,:],x_true[i5,:],plot_type='state',z=z_q5,ax=axs[1])
    axs[0].set_title("Steady State Covariance & Gain KF: 1st Mass")
    axs[1].set_title("Steady State Covariance & Gain KF: 5th Mass")
    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pD[i1,:], Pxhat_pD[i1,i1,:],x_true[i1,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_pD[i5,:], Pxhat_pD[i5,i5,:],x_true[i5,:],plot_type='error',z=z_q5,ax=axs[1])
    axs[0].set_ylim([-0.1,0.1])
    axs[1].set_ylim([-0.2,0.2])
    axs[0].set_title("Steady State Covariance & Gain KF: 1st Mass Error")
    axs[1].set_title("Steady State Covariance & Gain KF: 5th Mass Error")
    fig.tight_layout()
    plt.show()
