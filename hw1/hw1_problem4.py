"""
MAE 6760 Model Based Estimation
Homework #1
Random processes (car)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from scipy.stats import norm
import control

# -------------------------------------------------
# Constants
# -------------------------------------------------
M = 300
m = 50
K1 = 3000
K2 = 30000
C1 = 600

# -------------------------------------------------
# Open-loop state-space model
# x = [z_s, z_u, z_s_dot, z_u_dot]
# -------------------------------------------------
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-K1/M,  K1/M,  -C1/M,  C1/M],
    [ K1/m, -(K1+K2)/m,  C1/m, -C1/m]
])

Bu = np.array([[0], [0], [1/M], [-1/m]])     # actuator input
Br = np.array([[0], [0], [0], [K2/m]])       # road disturbance

Cz = np.array([[1, 0, 0, 0]])                # output: driver position
Dzu = np.array([[0]])
Dzr = np.array([[0]])

# Road disturbance intensity
Sigr = 2e-4

# -------------------------------------------------
# Part (a): simulate open-loop system
# -------------------------------------------------
Tf = 1000
dt = 0.01
t = np.arange(0, Tf + dt, dt)

np.random.seed(1)
r = np.sqrt(Sigr) * np.random.randn(len(t)) / np.sqrt(dt)



# -------------------------------------------------
# Part (b): closed-loop LQR design and simulation
# -------------------------------------------------
Rzz = 1
Ruu = 2e-9

# LQR (same role as lqry in MATLAB)
Q = Cz.T @ Cz * Rzz
R = np.array([[Ruu]])

K, S, E = control.lqr(A, Bu, Q, R)
K = np.asarray(K)


# -------------------------------------------------
# Part (c): plot open vs closed loop
# -------------------------------------------------


# -------------------------------------------------
# Part (d): analyze control effort
# -------------------------------------------------

