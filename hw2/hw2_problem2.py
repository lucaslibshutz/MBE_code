import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import triang, uniform
# import sympy as sp
from sympy import solve, symbols, N

np.set_printoptions(precision=5, suppress=True)

# Define triangular PDF
pdv = triang(c=0.0, loc=0.0, scale=1.0)

mn_v = float(pdv.mean())
sig_v = float(pdv.std())

v_stats = np.array([mn_v, sig_v])

print("mean and std of the sensor noise are:")
print(v_stats)

v = np.arange(-1.5, 1.5 + 0.001, 0.001)
fv = pdv.pdf(v)

# plt.figure(1)
# plt.plot(v, fv, 'b-')
# plt.ylabel(r'Sensor Noise Model $f(v)$')
# plt.xlabel('v')
# plt.grid(True)
#
# Define uniform prior PDF
pdx0 = uniform(loc=3.0, scale=1.0)

min_x0 = pdx0.mean()
sig_x0 = pdx0.std()

x0_stats = np.array([min_x0, sig_x0])

print("\n Mean and std of the prior are:")
print(x0_stats)

x = np.arange(2, 5+0.001,0.001)
fx0 = pdx0.pdf(x)
#
# plt.figure(2)
# plt.plot(x,fx0,'b-')
# plt.ylabel(r'Prior Distribution $f(x_0)$')
# plt.xlabel(r'$x_0$')
# plt.grid(True)


print(f"{np.cov(fx0):.6f}")
