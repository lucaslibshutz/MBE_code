import numpy as np
from MMSEestimate import mmse_estimate_batch
# from matplotlib import pyplot as plt

np.set_printoptions(precision=8, suppress=True)
def print_arr_sci(arr):
  new_arr = np.array2string(arr, formatter={'float_kind':'{:.4e}'.format})
  print(new_arr)


t = np.arange(0.1, 1.0 + 0.1, 0.1).reshape(-1,1)
n = len(t)

x = np.ones(3).reshape(-1,1)

R_diag = np.array([
    0.001, 0.002, 0.005, 0.010, 0.008,
    0.002, 0.010, 0.007, 0.020, 0.006
])**2

R = np.diag(R_diag)

v = np.linalg.cholesky(R) @ np.random.randn(n,1)

z = (
    x[0]
    + x[1] * np.sin(10* t)
    + x[2] * np.exp(2 * t**2)
    + v
)

# Prior information
x0 = np.array([1.01, 0.98, 0.99]).reshape(-1,1)

P0 = np.eye(3) * 0.001

# Part (a): MMSE Estimate
# Assemble H as function of time
H = np.column_stack([
    np.ones_like(t),
    np.sin(10*t),
    np.exp(2*t**2)
])

# Do MMSE in batched form

xhatA, PxhatA = mmse_estimate_batch(z, H, R, x0, P0)
print(xhatA)
print_arr_sci(PxhatA)

# Part (b): Varying sensor noise

# Define alpha array
alpha = 10.0 ** np.arange(-3,4)

R_list = np.array([alpha_i * R for alpha_i in alpha])

xhatB_list = []
PxhatB_list = []
for Rv in R_list:
    xhatB, PxhatB = mmse_estimate_batch(z, H, Rv, x0, P0)
    xhatB_list.append(xhatB)
    PxhatB_list.append(PxhatB)
xhatB_list = np.array(xhatB_list)
PxhatB_list = np.array(PxhatB_list)

# Part (c): Varying prior

Prior_list = np.array([alpha_i * P0 for alpha_i in alpha])
print(np.shape(Prior_list))
xhatC_list = []
PxhatC_list = []
for prior in Prior_list:
    xhatC, PxhatC = mmse_estimate_batch(z, H, R, x0, prior)
    xhatC_list.append(xhatC)
    PxhatC_list.append(PxhatC)

xhatC_list = np.array(xhatC_list)
PxhatC_list = np.array(PxhatC_list)

# Need to make visualizations of matrix entries to show how
# noise affects covariance, and then show how accurate estimation is for different \alpha values.
