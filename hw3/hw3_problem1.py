import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import addcopyfighandler

from plot_openloop import plot_openloop
from plot_estimator import plot_estimator

# environment vars
plot_A = True
plot_D = False

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

# Initialize variables
xhat_pA = np.zeros((nx,nk))
xhat_uA = np.zeros((nx,nk))
xhat_pA[:,0] = x0 # initial state estimate
xhat_uA[:,0] = x0 # initial state estimate
# xhat_pA[:,0] = np.zeros(nx) # initial state estimate
# xhat_uA[:,0] = np.zeros(nx) # initial state estimate
Pxhat_pA = np.zeros((nx,nx,nk))
Pxhat_uA = np.zeros((nx,nx,nk))
Pxhat_pA[:,:,0] = np.eye(nx,nx) # initial error covariance
Pxhat_uA[:,:,0] = np.eye(nx,nx) # initial error covariance
Q = np.atleast_2d(Qsim) # process noise covariance
# stack measurements into single array
zA = np.vstack((z_q1,z_q5))

for k in range(nk-1):
    # Prediction step
    xhat_pA[:,k+1] = F @ xhat_uA[:,k]
    Pxhat_pA[:,:,k+1] = F @ Pxhat_uA[:,:,k] @ F.T + G @ Q @ G.T

    xPred = xhat_pA[:,k+1]
    PxPred = Pxhat_pA[:,:,k+1]

    # Kalman Gain
    K_ins = Hc @ PxPred @ Hc.T + np.diag([Rq1, Rq5])
    K = (PxPred @ Hc.T)@ np.linalg.inv(K_ins)

    # Update step
    xhat_uA[:,k+1] = xPred + K @ (zA[:,k+1]- Hc@ xPred)
    # PxhatA[:, :, k+1] = (np.eye(nx) - K @ Hc) @ PxPred
    origTerm = np.eye(nx) - K @ Hc
    Pxhat_uA[:, :, k+1] = origTerm @ PxPred @ origTerm.T + K @ np.diag([Rq1, Rq5]) @ K.T

if plot_A:
    # Plot results
    fig, axs = plt.subplots(1,2, figsize=(16,6))

    # First mass
    # axs[0].plot(t[::2], z_q1[::2], 'g.', label='Measurements')
    plot_estimator(t, xhat_uA[i1,:], Pxhat_uA[i1,i1,:],x_true[i1,:],plot_type='state',z=z_q1,ax=axs[0])
    axs[0].set_ylabel(r"position of first mass $m_1$")

    # Fifth mass
    # axs[1].plot(t[::2], z_q5[::2], 'g.', label='Measurements')
    plot_estimator(t, xhat_uA[i5,:], Pxhat_uA[i5,i5,:],x_true[i5,:],plot_type='state',z=z_q5,ax=axs[1])
    axs[1].set_ylabel(r"position of fifth mass $m_5$")
    fig.tight_layout()
    plt.show()
    ### --------------------------------------
    ### (b) Predicted vs Updated Covariances for mass 5
    ### --------------------------------------
    fig,axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pA[i1,:], Pxhat_pA[i1,i1,:],x_true[i1,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_uA[i1,:], Pxhat_uA[i1,i1,:],x_true[i1,:],plot_type='error',z=z_q1,ax=axs[1])
    axs[0].set_title(r"Predicted Error $(q_1)$")
    axs[1].set_title(r"Updated Error $(q_1)$")
    fig.tight_layout()
    plt.show()

    fig,axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_pA[i5,:], Pxhat_pA[i5,i5,:],x_true[i5,:],plot_type='error',ax=axs[0])
    plot_estimator(t, xhat_uA[i5,:], Pxhat_uA[i5,i5,:],x_true[i5,:],plot_type='error',z=z_q5,ax=axs[1])
    axs[0].set_title(r"Predicted Error $(q_5)$")
    axs[1].set_title(r"Updated Error $(q_5)$")
    fig.tight_layout()
    plt.show()


    # Plot variances and error of fifth mass for predicted and updated
    fig, axs = plt.subplots(1,2, figsize=(16,6))

    axs[0].semilogy(t, Pxhat_pA[i1,i1,:], 'r',label='Predicted')
    axs[0].semilogy(t, Pxhat_uA[i1,i1,:], 'b',label='Updated')
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel(r"$\sigma^2(m_1)$")
    axs[0].legend()

    axs[1].semilogy(t, Pxhat_pA[i5,i5,:],'r', label='Predicted')
    axs[1].semilogy(t, Pxhat_uA[i5,i5,:],'b', label='Updated')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel(r"$\sigma^2(m_5)$")
    axs[1].legend()
    fig.tight_layout()
    plt.show()

### --------------------------------------
### (c) Steady State Error Covariance
### --------------------------------------
Pbar = solve_discrete_are(a=F.T, b=Hc.T, q=G @ Q @ G.T, r=np.diag([Rq1, Rq5]))

print(f"Simulated measurement covariance at the final time is: {Pxhat_uA[i5,i5,nk-1]}")
print(f"Steady state error covariance is: {Hq5 @ Pbar @ Hq5.T}")


### --------------------------------------
### (d) Constant gain KF
### --------------------------------------
# Perform the same KF from (a), but this time use P0 = Pbar

# Initialize variables
xhat_pD = np.zeros((nx,nk))
xhat_uD = np.zeros((nx,nk))
xhat_pD[:,0] = x0 # initial state estimate
xhat_uD[:,0] = x0 # initial state estimate
# xhat_pD[:,0] = np.zeros(nx) # initial state estimate
# xhat_uD[:,0] = np.zeros(nx) # initial state estimate
Pxhat_pD = np.zeros((nx,nx,nk))
Pxhat_uD = np.zeros((nx,nx,nk))
Pxhat_pD[:,:,0] = Pbar # initial error covariance
Pxhat_uD[:,:,0] = Pbar # initial error covariance
Q = np.atleast_2d(Qsim) # process noise covariance
# stack measurements into single array
zD = np.vstack((z_q1,z_q5))

for k in range(nk-1):
    # Prediction step
    xhat_pD[:,k+1] = F @ xhat_uD[:,k]
    Pxhat_pD[:,:,k+1] = F @ Pxhat_uD[:,:,k] @ F.T + G @ Q @ G.T

    xPred = xhat_pD[:,k+1]
    PxPred = Pxhat_pD[:,:,k+1]

    # Kalman Gain
    K_ins = Hc @ PxPred @ Hc.T + np.diag([Rq1, Rq5])
    K = (PxPred @ Hc.T)@ np.linalg.inv(K_ins)

    # Update step
    xhat_uD[:,k+1] = xPred + K @ (zD[:,k+1]- Hc@ xPred)
    # PxhatD[:, :, k+1] = (np.eye(nx) - K @ Hc) @ PxPred
    origTerm = np.eye(nx) - K @ Hc
    Pxhat_uD[:, :, k+1] = origTerm @ PxPred @ origTerm.T + K @ np.diag([Rq1, Rq5]) @ K.T

# Plot estimator for part (D)
if plot_D:
    fig, axs = plt.subplots(1,2, figsize=(16,6))
    plot_estimator(t, xhat_uD[i1,:], Pxhat_uD[i1,i1,:],x_true[i1,:],plot_type='state',z=z_q1,ax=axs[0])
    plot_estimator(t, xhat_uD[i5,:], Pxhat_uD[i5,i5,:],x_true[i5,:],plot_type='state',z=z_q5,ax=axs[1])
    axs[0].set_title("Steady State Covariance KF: 1st Mass")
    axs[1].set_title("Steady State Covariance KF: 5th Mass")
    fig.tight_layout()
    plt.show()
