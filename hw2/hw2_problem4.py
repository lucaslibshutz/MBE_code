import numpy as np
from MMSEestimate import mmse_estimate_batch
from matplotlib import pyplot as plt
import addcopyfighandler

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

print(xhatB_list.shape)
print(PxhatB_list.shape)

# Plot results

xhat = xhatB_list
P = PxhatB_list

N_alpha, state_dim = xhat.shape

xi = xhat.T
sigi = np.array([np.sqrt(np.diag(P[a])) for a in range(N_alpha)]).T

fig, axes = plt.subplots(1, state_dim, figsize=(12,4),sharex=True)
true_val = np.ones_like(alpha)

for i in range(state_dim):
    axes[i].errorbar(alpha, xi[i, :], yerr=2 * sigi[i, :], fmt='o-', capsize=3, label='est ± 2σ')
    axes[i].plot(alpha, true_val, linestyle='--', label='true value')
    # only use log scale if alphai > 0
    if np.all(alpha > 0):
        axes[i].set_xscale('log')
    axes[i].grid(True)
    axes[i].set_xlabel('scale factor alpha')
    axes[i].set_ylabel(f'state {i} estimate')

axes[0].legend(loc='lower left')
fig.suptitle(r'Study of sensor noise scale factor $\alpha$ ($R=\alpha R_0$)')
plt.tight_layout()

# Part (c): Varying prior

Prior_list = np.array([alpha_i * P0 for alpha_i in alpha])
xhatC_list = []
PxhatC_list = []
for prior in Prior_list:
    xhatC, PxhatC = mmse_estimate_batch(z, H, R, x0, prior)
    xhatC_list.append(xhatC)
    PxhatC_list.append(PxhatC)

xhatC_list = np.array(xhatC_list)
PxhatC_list = np.array(PxhatC_list)

# Plot part (c)
xhat = xhatC_list
P = PxhatC_list

xi = xhat.T
sigi = np.array([np.sqrt(np.diag(P[a])) for a in range(N_alpha)]).T
# Plot
fig, axes = plt.subplots(1, state_dim, figsize=(12, 4), sharex=True)
# change true_val if your true state isn't ones
true_val = np.ones_like(alpha)

for i in range(state_dim):
    axes[i].errorbar(alpha, xi[i, :], yerr=2 * sigi[i, :], fmt='o-', capsize=3, label='est ± 2σ')
    axes[i].plot(alpha, true_val, linestyle='--', label='true value')
    if np.all(alpha > 0):
        axes[i].set_xscale('log')
    axes[i].grid(True)
    axes[i].set_xlabel('scale factor (prior) α')
    axes[i].set_ylabel(f'state {i} estimate')

axes[0].legend(loc='lower left')
fig.suptitle(r'Study of prior scale factor $\alpha$ (Prior = $\alpha P_0$)')
plt.tight_layout()
plt.show()


# Need to make visualizations of matrix entries to show how
# noise affects covariance, and then show how accurate estimation is for different \alpha values.
