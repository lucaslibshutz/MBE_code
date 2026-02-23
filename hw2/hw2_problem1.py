#%%
import numpy as np
from matplotlib import pyplot as plt
from lsEstimate import LSestimate

def print_arr_sci(arr):
  new_arr = np.array2string(arr, formatter={'float_kind':'{:.4e}'.format})
  print(new_arr)

# Define measurement model
# z_k = H x + v
#
H = np.array([
    [1, -1],
    [10, 1],
    [1, 10],
    ],
dtype=float)

xtrue = np.array([2.0,10.0]).reshape(2,1)
n=  20
#%%

# Get repeatable random seed
np.random.seed(14)
np.set_printoptions(precision=6, suppress=True)

# Generate measurement data
#
sig1 = 10
sig2 = 1
sig3 = 100

va = np.diag([sig1, sig2, sig3]) @ np.random.randn(3,n)
za = np.tile(H @ xtrue, (1,n)) + va

# Part (a): LS estimate using noisy data z
#%%
xhatA, PxhatA = LSestimate(za, H)
xhatA = xhatA.reshape((2,1)) # convert to column vector

print(xhatA)
print(PxhatA)
# %%

# Part (b): Weighted LS (known noise covariance)
zb = za.copy()
Rvb = np.diag([10**2, 1**2, 100**2])

xhatB, PxhatB = LSestimate(zb, H, Rvb)
xhatB = xhatB.reshape((2,1))

print(xhatB)
print(PxhatB)

# %%
# Part (c): Theoretical LS estimator covariance

P_xtA = (1/n) * np.linalg.inv(np.transpose(H) @ np.linalg.inv(np.eye(za.shape[0])) @ H)
P_xtB = (1/n) * np.linalg.inv(np.transpose(H) @ np.linalg.inv(Rvb) @ H)

print(P_xtA)
print(P_xtB)


# %%
# Part (d): Noise covariance estimate
rHat_va = np.zeros((3,3))
for k in range(n):
    zk = za[:,[k]]
    outMat = zk - H @ xhatA
    rHat_va += zk - H @ xhatA
rHat_va = rHat_va / n
print("rHat_va:",rHat_va)

# Put this into new LS estimate
xhatD, PxhatD = LSestimate(za, H, rHat_va)
xhatD = xhatD.reshape((2,1))
print(xhatD)
print(PxhatD)

# %%
# Part (e): z_3 with uniform distribution
Ue = 173

v3e = Ue * (2 * (np.random.rand(1,n) - 0.5))
ve = np.vstack((va[0,:], va[1,:], v3e))

ze = np.tile(H @ xtrue, (1,n)) + ve

Rve = np.diag([sig1**2, sig2**2, np.cov(v3e)**2])

xhatE, PxhatE = LSestimate(ze, H, Rve)
xhatE = xhatE.reshape((2,1))

print(xhatE)
print(PxhatE)
