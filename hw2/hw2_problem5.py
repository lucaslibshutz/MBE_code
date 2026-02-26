import numpy as np
from nonlinearLS import nonlinearLS
from beaconNLS import beaconNLS

np.set_printoptions(precision=5, suppress=True)
np.random.seed(10)

# Problem setup: beacons
bA = np.array([-10.0,100.0]).reshape(-1,1)
bB = np.array([490.0,20.0]).reshape(-1,1)
bC = np.array([500.0,40.0]).reshape(-1,1)

xtrue = np.array([-5.0, 2.0]).reshape(-1,1)

# Perfect ranges
RA = np.linalg.norm(bA - xtrue)
RB = np.linalg.norm(bB - xtrue)
RC = np.linalg.norm(bB - xtrue)

n = 10 # Number of measurements

# Part (a): Three beacons (A,B,C)

R_a = np.diag(10.0 * np.ones(3))
v_a = np.linalg.cholesky(R_a) @ np.random.randn(3,n)
z_a = np.tile(np.array([RA, RB, RC]).reshape(-1,1), (1,n)) + v_a

x0 = np.array([0.0,0.0])
xS = np.vstack((bA.T,bB.T,bC.T))

xhatA, PxhatA, HhatA, iter_countA = beaconNLS(z_a, R_a, x0, xS)

print(xhatA)
print(iter_countA)

# Part (b): Two beacons (A,B) and (B,C)

# (A,B)
ii = [0,1]
R_b1 = R_a[np.ix_(ii,ii)]
z_b1 = z_a[ii,:]

# (B,C)
ii = [1,2]
R_b2 = R_a[np.ix_(ii,ii)]
z_b2 = z_a[ii,:]

# Part (c): Perfect linearization

# Part (d): Correlated noise
R_d = np.array([
    [10, 0, 0],
    [0, 10, 9],
    [0, 9, 10]],dtype=float)

v_d = np.linalg.cholesky(R_d) @ np.random.randn(3,n)
z_d = np.tile(np.array([RA, RB, RC]).reshape(-1,1), (1,n)) + v_d

