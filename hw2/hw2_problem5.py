import numpy as np
from beaconNLS import beaconNLS

np.set_printoptions(precision=5, suppress=True)
np.random.seed(20)

# Problem setup: beacons
bA = np.array([-10.0,100.0]).reshape(-1,1)
bB = np.array([490.0,20.0]).reshape(-1,1)
bC = np.array([500.0,40.0]).reshape(-1,1)

xtrue = np.array([-5.0, 2.0]).reshape(-1,1)

# Perfect ranges
RA = np.linalg.norm(bA - xtrue)
RB = np.linalg.norm(bB - xtrue)
RC = np.linalg.norm(bC - xtrue)

n = 10 # Number of measurements

# Part (a): Three beacons (A,B,C)
print("Problem 5(a): Three beacons (A,B,C)")

R_a = np.diag([10.0,10.0,10.0])
v_a = np.linalg.cholesky(R_a) @ np.random.randn(3,n)
z_a = np.tile(np.array([RA, RB, RC]).reshape(-1,1), (1,n)) + v_a

x0 = np.array([0.0,0.0])
# xS = np.vstack((bA.T,bB.T,bC.T))
xS = np.array([
    [-10.0,100.0],
    [490.0,20.0],
    [500.0,40.0]
])

xhatA, PxhatA, HhatA, iter_countA = beaconNLS(z_a, R_a, x0, xS)

print(f"xHat = {xhatA}")
print(f"Pxhat = {PxhatA}")
print(f"Number of iterations: {iter_countA}")

# Part (b): Two beacons (A,B) and (B,C)

# (A,B)
ii = [0,1]
R_b1 = R_a[np.ix_(ii,ii)]
z_b1 = z_a[ii,:]
#NOTE: Compute B and C together to avid defining vars ii, z_bi, R_bi twice

xhatB1, PxhatB1, HhatB1, iter_countB1 = beaconNLS(z_b1, R_b1, x0, xS[ii,:])
xhatC1, PxhatC1, HhatC1 = beaconNLS(z_b1, R_b1, x0, xS[ii,:],xtrue.flatten())
print("\nProblem 5(b): Two beacons (A,B)")
print(f"xHat = {xhatB1}")
print(f"Pxhat = {PxhatB1}")

# (B,C)
ii = [1,2]
R_b2 = R_a[np.ix_(ii,ii)]
z_b2 = z_a[ii,:]

xhatB2, PxhatB2, HhatB2, iter_countB2 = beaconNLS(z_b2, R_b2, x0, xS[ii,:])
xhatC2, PxhatC2, HhatC2 = beaconNLS(z_b2, R_b2, x0, xS[ii,:],xtrue.flatten())
print("\nProblem 5(b): Two beacons (B,C)")
print(f"xHat = {xhatB2}")
print(f"Pxhat = {PxhatB2}")

# Part (c): Perfect linearization
# Repeat two cases from (b) but with x0=xTrue


print("\nProblem 5(c): Perfect linearization")
print(f"(A,B) with x0=xTrue: xHat = {xhatC1}, Pxhat = {PxhatC1}")
print(f"(B,C) with x0=xTrue: xHat = {xhatC2}, Pxhat = {PxhatC2}")

# Part (d): Correlated noise
R_d = np.array([
    [10, 0, 0],
    [0, 10, 9],
    [0, 9, 10]],dtype=float)

v_d = np.linalg.cholesky(R_d) @ np.random.randn(3,n)
z_d = np.tile(np.array([RA, RB, RC]).reshape(-1,1), (1,n)) + v_d

xhatD, PxhatD, HhatD, iter_countD = beaconNLS(z_d, R_d, x0, xS)
print("\nProblem 5(d): Correlated Noise")
print(f"xHat = {xhatD}")
print(f"PxHat = {PxhatD}")
