"""
MAE 6760 Model Based Estimation
Cornell University
M Campbell

Homework #3
Problem #2: Kalman Filter
    Two aircraft tracking estimator in 2D as they turn
    uses the following Python functions:
        plot_openloop.py
        plot_estimator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
from scipy.stats import chi2
from scipy.linalg import sqrtm, block_diag
import addcopyfighandler #noqa F401

from plot_estimator import plot_estimator
from kalmanFilter import kalman_filter

np.random.seed(101)

## Environment vars for plotting
plot_I = 0
plot_A = 0
plot_B = 1
plot_C = 1
plot_D = 0
plot_E = 0

# ============================================================
# 2D Aircraft Open Loop Plot
# ============================================================

def plot_openloop_2D_aircraft(x_A, x_B=None):

    plt.figure(figsize=(8,6))

    plt.plot(x_A[0,:], x_A[2,:], 'b-',label='Aircraft A')
    plt.plot(x_A[0,0], x_A[2,0], 'b>')
    plt.plot(x_A[0,-1], x_A[2,-1], marker='*', color='b')

    if x_B is not None:
        plt.plot(x_B[0,:], x_B[2,:], 'r:',label='Aircraft B')
        plt.plot(x_B[0,0], x_B[2,0], 'r>')
        plt.plot(x_B[0,-1], x_B[2,-1], marker='*', color='r')

    plt.legend()

    plt.xlabel("East")
    plt.ylabel("North")
    plt.grid(True)


# ============================================================
# Aircraft A Model
# ============================================================

# continuous time model for aircraft A
Omega_A = 0.045
nx = 4
A_A = np.array([[0,1,0,0],
                [0,0,0,-Omega_A],
                [0,0,0,1],
                [0,Omega_A,0,0]])

B_A = np.array([[0,0],[1,0],[0,0],[0,1]])
H_A = np.array([[1,0,0,0],
                [0,0,1,0]])

# discrete time model for aircraft A
dt = 0.5
F_A, G_A, _, _, _ = cont2discrete((A_A,B_A,np.eye(4),0), dt)

Q_A = 10*np.array([[2.0,0.05],[0.05,0.5]])
R_A = np.array([[20,0.05],[0.05,20]])

tvec = np.arange(0,100+dt,dt)
nk = len(tvec)

x0_A = np.array([0,85*np.cos(np.pi/4),0,-85*np.sin(np.pi/4)])

# simulate aircraft A
w_A = sqrtm(Q_A) @ np.random.randn(2,nk)
x_A = np.zeros((4,nk))
x_A[:,0] = x0_A

for k in range(nk-1):
    x_A[:,k+1] = F_A @ x_A[:,k] + G_A @ w_A[:,k]

z_A = H_A @ x_A + sqrtm(R_A) @ np.random.randn(2,nk)

if plot_I:
    plot_openloop_2D_aircraft(x_A)
    plt.title("2D trajectory for aircraft A")
    plt.show()

# ============================================================
# (a) Baseline Kalman Filter
# ============================================================
# goal: KF to track/localize aircraft A
# 
# use model for aircraft A for this problem
# use measurements z_A for this problem
# 

## YOUR CODE HERE
xhatp_A, xhatu_A, Pp_A, Pu_A = kalman_filter(
    F = F_A,
    G = G_A,
    H = H_A,
    Q = Q_A,
    R = R_A,
    x0 = x0_A,
    P0 = np.eye(nx),
    z = z_A,
    nk = nk
)

if plot_A:
    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_A[0,:],Pu_A[0,0,:],x_A[0,:],plot_type='state',z=z_A[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    plot_estimator(tvec,xhatu_A[2,:],Pu_A[2,2,:],x_A[2,:],plot_type='state',z=z_A[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    plt.suptitle("(a) Baseline Kalman Filter for aircraft $A$: states", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_A[0,:],Pu_A[0,0,:],x_A[0,:],plot_type='error',z=z_A[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    plot_estimator(tvec,xhatu_A[2,:],Pu_A[2,2,:],x_A[2,:],plot_type='error',z=z_A[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    plt.suptitle("(a) Baseline Kalman Filter for aircraft $A$: errors", fontweight='bold')
    plt.tight_layout()
    plt.show()
# ============================================================
# (b) 95% Measurement Gating
# ============================================================
# goal: KF to track/localize aircraft A with measurement gate
# 
# use model for aircraft A for this problem
# use measurements z_A for this problem
# 
# variables to generate for display/plots:
#  Nrej: number of measurement rejections (subset of total nk)
#  Trej: times tvec[k+1] when msmt is rejected 
#  Erej: error z[:,k+1]-H*x_A(:,k+1] when msmt is rejected

# msmt gating parameters
Lam0 = chi2.ppf(0.95,2) # equivalent of chi2inv(0.95,2) in MATLAB
Lam = np.zeros(nk)

Nrej = 0
Irej = []

## YOUR CODE HERE
# Define innovatioon per time step, and then add to Nrej if rejected, and add index to Irej
for k in range(nk-1):
    inn = z_A[:,k+1] - H_A @ xhatp_A[:,k+1]
    S = H_A @ Pp_A[:,:,k+1] @ H_A.T + R_A

    Lam[k+1] = inn.T @ np.linalg.inv(S) @ inn

    if Lam[k+1] > Lam0:
        Nrej += 1
        Irej.append(k+1)

# uncomment lines below to output the percent of msmts rejected
print(f"(b) percent msmts rejected: {Nrej/nk*100:.3f}%")

if plot_B:
    fig, axs = plt.subplots(1,2,figsize=(16,6))

    Trej = tvec[Irej]
    Erej = z_A[:,Irej] - H_A @ x_A[:,Irej]

    plot_estimator(tvec,xhatu_A[0,:],Pu_A[0,0,:],x_A[0,:],plot_type='error',z=z_A[0,:],ax=axs[0])
    # uncomment the line below to add msmt rejections to the plot
    axs[0].plot(Trej, Erej[0,:], 'mo', fillstyle='none', label='Msmt Rejection')
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()

    plot_estimator(tvec,xhatu_A[2,:],Pu_A[2,2,:],x_A[2,:],plot_type='error',z=z_A[1,:],ax=axs[1])
    # uncomment the line below to add msmt rejections to the plot
    axs[1].plot(Trej, Erej[1,:], 'mo', fillstyle='none', label='Msmt Rejection')
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(b) Kalman Filter with 95% msmt gating for aircraft A", fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# (c) Process Noise Tuning + Filter Integrity
# ============================================================
# goal: use hypothesis tests to tune process noise covariance Q 
# 
# use model for aircraft A for this problem
# load simulated measurements from tile (these are for aircraft A, but labeled z_c for Part c
# use measurements z_c for this problem
# use states x_c for the plots on problem
# 
# variables to generate for display/plots:
#  Lam: innovations test statistic lambda (1 x nk)
#  LamF: Kalman Filter test statistic lambda (1 x nk)
#  Nrej: number of measurement rejections (subset of total nk)
#  Trej: times tvec[k+1] when msmt is rejected 
#  Erej: error z[:,k+1]-H*x_c[:,k+1] when msmt is rejected
#  NFrej: number of times the filter is inconsistent (subset of total nk)
#  TFrej: times tvec[k+1] when the filter is inconsistent 
#  EFrej: error z[:,k+1]-H*x_c[:,k+1] when the filter is inconsistent

#load measurement vector z_c and state x_c
data = np.load("ACdata.npz")
z_c = data["z_c"]
x_c = data["x_c"]

x0_C = x_c[:,0]
C_factor = 10
Q_C = C_factor * Q_A



Nrej = 0
Irej = []
NFrej = 0
IFrej = []

# msmt gating parameters
Lam0 = chi2.ppf(0.95,2)
Lam = np.zeros(nk)

# filter consistency parameters
Lam = np.zeros(nk)
LamF = np.zeros(nk)
win=10
Blow = chi2.ppf(0.025,10*2)/win
Bhigh = chi2.ppf(0.975,10*2)/win
#
# rerun new KF on this data
xhatp_C, xhatu_C, Pp_C, Pu_C = kalman_filter(
    F = F_A,
    G = G_A,
    H = H_A,
    Q = Q_C,
    R = R_A,
    x0 = x0_C,
    P0 = np.eye(nx),
    z = z_c,
    nk = nk,
    Lambda0 = Lam0 # adding in meaasurement rejection
)
## YOUR CODE HERE
for k in range(nk-1):
    inn = z_c[:,k+1] - H_A @ xhatp_C[:,k+1]
    S = H_A @ Pp_C[:,:,k+1] @ H_A.T + R_A

    Lam[k+1] = inn.T @ np.linalg.inv(S) @ inn

    # update measurement only if gating passes
    if Lam[k+1] > Lam0:
        Nrej += 1
        Irej.append(k+1)

    if k >= win:
        LamF[k+1] = np.mean(Lam[k-win+2:k+1])
    if (LamF[k+1] < Blow) or (LamF[k+1] > Bhigh):
        NFrej += 1
        IFrej.append(k+1)


# uncomment lines below to output the percent of msmts rejected
print("(c) percent msmts rejected:",Nrej/nk*100)
print("(c) percent time KF inconsistent:",NFrej/nk*100)

if plot_C:
    Trej = tvec[Irej]
    Erej = z_A[:,Irej] - H_A @ x_A[:,Irej]

    TFrej = tvec[IFrej]
    EFrej = z_A[:,IFrej] - H_A @ x_A[:,IFrej]


    fig, axs = plt.subplots(1,2,figsize=(16,6))

    plot_estimator(tvec,xhatu_C[0,:],Pu_C[0,0,:],x_c[0,:],plot_type='error',z=z_c[0,:],ax=axs[0])
    # uncomment the line below to add msmt rejections to the plot
    axs[0].plot(Trej, Erej[0,:], 'mo', fillstyle='none', label='Msmt Rejection')
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_C[2,:],Pu_C[2,2,:],x_c[2,:],plot_type='error',z=z_c[1,:],ax=axs[1])
    # uncomment the line below to add msmt rejections to the plot
    axs[1].plot(Trej, Erej[1,:], 'mo', fillstyle='none', label='Msmt Rejection')
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(c): Kalman Filter with Msmt Gate and Filter Integrity for aircraft A", fontweight='bold')
    plt.tight_layout()
    plt.show()

    plot_estimator(tvec,xhatu_C[0,:],Pu_C[0,0,:],x_c[0,:],plot_type='error',z=z_c[0,:],ax=axs[0])

    plt.figure(figsize=(8,6))
    plt.plot(tvec,Lam,'m:',label=r'inn test statistic $\lambda$')
    plt.plot(tvec[win+1:],LamF[win+1:],'-',color='purple',label=r'KF test statistic $\lambda_{10}^{KF}$')
    plt.plot(tvec,Blow*np.ones(nk),'b--',label='lower bound')
    plt.plot(tvec,Bhigh*np.ones(nk),'b--',label='upper bound')
    plt.ylabel('filter integrity')
    plt.xlabel('time $t$ (sec)')
    plt.legend()
    plt.title(f"(c) Kalman Filter integrity test statistic, for Q factor of {C_factor}", fontweight='bold')
    plt.show()


# ============================================================
# (d) Joint Kalman Filter (Aircraft A & B)
# ============================================================
# goal: estimate X=[x_A;x_B] given Z=[z_A;z_B]
# 
# use both models for aircraft A and B for this problem
# use measurements Z_A and Z_B for this problem

# continuous time model for aircraft B
Omega_B = -0.045
A_B = np.array([[0,1,0,0],
                [0,0,0,-Omega_B],
                [0,0,0,1],
                [0,Omega_B,0,0]])
B_B = np.array([[0,0],[1,0],[0,0],[0,1]])

# discrete time model for aircraft B
F_B, G_B, _, _, _ = cont2discrete((A_B,B_B,np.eye(nx),0), dt)
Q_B = Q_A
R_B = R_A
H_B = H_A

# simulate aircraft B
x0_B = np.array([4000,85*np.cos(np.pi/4),3200,-85*np.sin(np.pi/4)])

x_B = np.zeros((4,nk))
x_B[:,0]=x0_B
for k in range(nk-1):
    x_B[:,k+1] = F_B@x_B[:,k] + G_B@w_A[:,k]

z_B = H_B@x_B + sqrtm(R_B)@np.random.randn(2,nk)

F_totD = block_diag(F_A,F_B)
G_totD = block_diag(G_A,G_B)
H_totD = block_diag(H_A,H_B)
Q_totD = block_diag(Q_A,Q_B)
R_totD = block_diag(R_A,R_B)
x0_totD = np.hstack((x0_A,x0_B))
z_totD = np.vstack((z_A,z_B))

print(f"F_totD shape: {F_totD.shape}")
print(f"G_totD shape: {G_totD.shape}")
print(f"H_totD shape: {H_totD.shape}")
print(f"Q_totD shape: {Q_totD.shape}")
print(f"R_totD shape: {R_totD.shape}")
print(f"x0_totD shape: {x0_totD.shape}")
print(f"z_totD shape: {z_totD.shape}")
print("nk:", nk)


## YOUR CODE HERE
xhatp_D, xhatu_D, Pp_D, Pu_D = kalman_filter(
    F = F_totD,
    G = G_totD,
    H = H_totD,
    Q = Q_totD,
    R = R_totD,
    x0 = x0_totD,
    P0 = np.eye(2*nx),
    z = z_totD,
    nk = nk
)


if plot_D:
    plot_openloop_2D_aircraft(x_A,x_B)
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_D[0,:],Pu_D[0,0,:],x_A[0,:],plot_type='error',z=z_A[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_D[2,:],Pu_D[2,2,:],x_A[2,:],plot_type='error',z=z_A[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(d): Joint Kalman Filter: aircraft A errors", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_D[nx+0,:],Pu_D[nx+0,nx+0,:],x_B[0,:],plot_type='error',z=z_B[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_D[nx+2,:],Pu_D[nx+2,nx+2,:],x_B[2,:],plot_type='error',z=z_B[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(d): Joint Kalman Filter: aircraft B errors", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_D[0,:],Pu_D[0,0,:],x_A[0,:],plot_type='state',z=z_A[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_D[2,:],Pu_D[2,2,:],x_A[2,:],plot_type='state',z=z_A[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(d): Joint Kalman Filter: aircraft A state", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_D[nx+0,:],Pu_D[nx+0,nx+0,:],x_B[0,:],plot_type='state',z=z_B[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_D[nx+2,:],Pu_D[nx+2,nx+2,:],x_B[2,:],plot_type='state',z=z_B[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(d): Joint Kalman Filter: aircraft B state", fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"Final covaraince matrix: {Pu_D[:,:,-1]}")


# ============================================================
# (e) Joint Kalman Filter with Relative Range Sensor
# ============================================================
# goal: estimate X=[X_A;X_B] given Z=[z_A;z_B]
#       --> repeat (d), but for a 2D range measurement
# 
# use both models for aircraft A and B for this problem
# use measurements Zr for this problem

Rr = np.array([[10,0.15],[0.15,10]])
Hr = np.array([[1,0,0,0,-1,0,0,0],
               [0,0,1,0,0,0,-1,0]])

Zr = np.vstack((x_A[0,:]-x_B[0,:],
                x_A[2,:]-x_B[2,:])) + sqrtm(Rr)@np.random.randn(2,nk)


# we only will have a range measurement, so:
xhatp_E, xhatu_E, Pp_E, Pu_E = kalman_filter(
    F = F_totD,
    G = G_totD,
    H = Hr,
    Q = Q_totD,
    R = Rr,
    x0 = x0_totD,
    P0 = np.eye(2*nx),
    z = Zr,
    nk = nk

)
## YOUR CODE HERE


if plot_E:
    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_E[0,:],Pu_E[0,0,:],x_A[0,:],plot_type='error',z=z_A[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_E[2,:],Pu_E[2,2,:],x_A[2,:],plot_type='error',z=z_A[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(e): Joint KF with Relative Range: aircraft A errors", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_E[nx+0,:],Pu_E[nx+0,nx+0,:],x_B[0,:],plot_type='error',z=z_B[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_E[nx+2,:],Pu_E[nx+2,nx+2,:],x_B[2,:],plot_type='error',z=z_B[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(e): Joint KF with Relative Range: aircraft B errors", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_E[0,:],Pu_E[0,0,:],x_A[0,:],plot_type='state',z=z_A[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_E[2,:],Pu_E[2,2,:],x_A[2,:],plot_type='state',z=z_A[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(e): Joint KF with Relative Range: aircraft A state", fontweight='bold')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2,figsize=(16,6))
    plot_estimator(tvec,xhatu_E[nx+0,:],Pu_E[nx+0,nx+0,:],x_B[0,:],plot_type='state',z=z_B[0,:],ax=axs[0])
    axs[0].set_ylabel("North error estimate $e_N(t)$")
    axs[0].legend()
    plot_estimator(tvec,xhatu_E[nx+2,:],Pu_E[nx+2,nx+2,:],x_B[2,:],plot_type='state',z=z_B[1,:],ax=axs[1])
    axs[1].set_ylabel("East error estimate $e_E(t)$")
    axs[1].legend()
    plt.suptitle("(e): Joint KF with Relative Range: aircraft B state", fontweight='bold')
    plt.tight_layout()
    plt.show()
