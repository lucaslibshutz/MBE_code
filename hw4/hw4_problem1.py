"""
MAE 6760 Model Based Estimation
Cornell University
M Campbell

Solution
Homework #4
Problem #1: Extended Kalman Filter (EKF)
    planar pose of a car with IMU biases
    uses the following Python functions:
        plot_estimator.py
        calculate_ellipse.py
"""

import numpy as np
import matplotlib.pyplot as plt

from plot_estimator import plot_estimator
from calculate_ellipse import calculate_ellipse

# -------------------------------------------------
# User input parameters
# -------------------------------------------------
# scenario_type = "baseline"
scenario_type = "swervy"

np.random.seed(100)

# ------------------------------------------------------------
# System model functions
# ------------------------------------------------------------

def predict_state_carposebias(Xk, U, dt):

    ##YOUR CODE HERE

    return Xkp1


def getFG_carposebias(X, dt):

    ##YOUR CODE HERE

    return F, G


# ------------------------------------------------------------
# Control input generator
# ------------------------------------------------------------

def get_controlinputs(scenario_type):

    if scenario_type == "baseline":

        Uacc = np.concatenate([
            np.ones(60),
            np.zeros(100),
            -np.ones(60),
            np.zeros(20),
            np.ones(60),
            np.zeros(50)
        ])

        Uomega = np.concatenate([
            np.zeros(240),
            -np.ones(40)*np.pi/2/4,
            np.zeros(70)
        ])

    elif scenario_type == "swervy":

        Uacc = np.concatenate([
            np.ones(60),
            np.zeros(230),
            -np.ones(60)
        ])

        Uomega = np.concatenate([
            np.zeros(60),
            np.ones(10)*np.pi/2/2,
            -np.ones(20)*np.pi/2/2,
            np.ones(20)*np.pi/2/2,
            -np.ones(20)*np.pi/2/2,
            np.ones(20)*np.pi/2/2,
            -np.ones(10)*np.pi/2/2,
            np.zeros(30),
            np.ones(10)*np.pi/2/2,
            -np.ones(20)*np.pi/2/2,
            np.ones(20)*np.pi/2/2,
            -np.ones(20)*np.pi/2/2,
            np.ones(20)*np.pi/2/2,
            -np.ones(10)*np.pi/2/2,
            np.zeros(60)
        ])

    else:
        raise ValueError("unknown scenario")

    return Uacc, Uomega

# ------------------------------------------------------------
# Simulate system
# ------------------------------------------------------------

Uacc, Uomega = get_controlinputs(scenario_type)

dt = 0.1
nt = len(Uacc)
t = np.arange(0, dt*nt, dt)

nx = 4
x_true = np.zeros((nx, nt))

for k in range(nt-1):

    Vk = x_true[2,k]
    Tk = x_true[3,k]

    x_true[:,k+1] = x_true[:,k] + dt*np.array([
        Vk*np.cos(Tk),
        Vk*np.sin(Tk),
        Uacc[k],
        Uomega[k]
    ])

# ---------------------------------------------------------
# Plot Birds-eye view of true trajectory
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(x_true[0,:], x_true[1,:], 'm.', linewidth=3)
plt.grid(True)
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.tight_layout()

# ---------------------------------------------------------
# Plot true velocity and heading
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(t, x_true[2,:], 'b-')
plt.plot(t, x_true[3,:], 'b--')
plt.xlabel("time (sec)")
plt.ylabel("ideal state")
plt.legend(["velocity (m/sec)", "heading (rad)"], loc="lower left")
plt.axis([0, nt*dt, -2, 6.5])
plt.grid(True)
plt.tight_layout()

# ------------------------------------------------------------
# Generate noise and measurements
# ------------------------------------------------------------

bias_acc = 0.1
bias_rg  = -0.025

x_true = np.vstack((x_true,
                    bias_acc*np.ones(nt),
                    bias_rg*np.ones(nt)))

nx = 6

Q = np.diag([0.1**2, 0.04**2])
w = np.linalg.cholesky(Q) @ np.random.randn(2,nt)

Zacc = Uacc + bias_acc + w[0,:]
Zrg  = Uomega + bias_rg + w[1,:]

nz = 2
R = np.eye(nz)

v = np.linalg.cholesky(R) @ np.random.randn(nz,nt)
z = x_true[0:2,:] + v

H = np.hstack((np.eye(2), np.zeros((2,4))))


# ------------------------------------------------------------
# Extended Kalman Filter (EKF)
# ------------------------------------------------------------

    ##YOUR CODE HERE

# ---------------------------------------------------------
# Estimator plots
# ---------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(16,6))
plot_estimator(t, xhatu[0,:], Pu[0,0,:], x_true[0,:],
               "error", z[0,:], ax=ax[0])
ax[0].set_ylabel("x position (m)")
plot_estimator(t, xhatu[1,:], Pu[1,1,:], x_true[1,:],
               "error", z[1,:], ax=ax[1])
ax[1].set_ylabel("y position (m)")
plt.tight_layout()

fig, ax = plt.subplots(1,2, figsize=(16,6))
plot_estimator(t, xhatu[2,:], Pu[2,2,:], x_true[2,:],
               "error", ax=ax[0])
ax[0].set_ylabel("V velocity (m/sec)")
thC = 180/np.pi
plot_estimator(t, xhatu[3,:]*thC, Pu[3,3,:]*thC**2, x_true[3,:]*thC,
               "error", ax=ax[1])
ax[1].set_ylabel(r"$\theta$ heading (deg)")
plt.tight_layout()

fig, ax = plt.subplots(1,2, figsize=(16,6))
plot_estimator(t, xhatu[4,:], Pu[4,4,:], x_true[4,:],
               "error", ax=ax[0])
ax[0].set_ylabel("accel bias (m/sec^2)")
plot_estimator(t, xhatu[5,:], Pu[5,5,:], x_true[5,:],
               "error", ax=ax[1])
ax[1].set_ylabel("RG bias (rad/sec)")
plt.tight_layout()

# ---------------------------------------------------------
# Trajectory with uncertainty ellipses
# ---------------------------------------------------------

plt.figure(figsize=(8,6))

pt, = plt.plot(x_true[0,:], x_true[1,:], 'm.')
ph, = plt.plot(xhatu[0,:], xhatu[1,:], 'b-')

iell = np.concatenate(([2], np.arange(10, nt, 10)))

patch_handles = []

for ii in iell:

    Xe, Ye, _, _, _ = calculate_ellipse(
        xhatu[0:2,ii],
        Pu[0:2,0:2,ii],
        3, 50
    )

    plt.plot(xhatu[0,ii], xhatu[1,ii], 'bx')

    ne = len(Xe)

    x1 = np.concatenate((Xe[int(ne/2):], Xe[:int(ne/2)]))
    y1 = np.concatenate((Ye[int(ne/2):], Ye[:int(ne/2)]))

    plt.plot(x1, y1, 'b-', label='1-sigma ellipse',linewidth=0.5)
    p = plt.fill(x1, y1, alpha=0.1, color='b')
    patch_handles.append(p[0])

plt.xlabel("x position (m)")
plt.ylabel("y position (m)")

plt.legend([ph, patch_handles[0], pt],
           ["position estimate", "error ellipse", "truth"],
           loc="lower left")

plt.tight_layout()

plt.show()