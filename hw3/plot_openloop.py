"""
plot_openloop(t, x_true, z=None, x_no_w=None):
This function plots a scalar true state along with
the measurement (optional) and state with no noise (w=0)
    Note: assumes scalar state

INPUTS
    t = time vector (1 x n)
    x_true = vector of the true state (1 x n)
    z = measurement vector (1 x n) (optional)
    x_no_w = vector of the state without noise (1 x n) (optional)

MAE 6760 Model Based Estimation
Cornell University
M Campbell
"""
import numpy as np
from matplotlib import pyplot as plt

MCcolors = {
    "blue": np.array([4, 51, 255]) / 255,
    "green": np.array([0, 160, 0]) / 255,
    "mag": np.array([255, 64, 255]) / 255,
}

def plot_openloop(t, x_true, z=None, x_no_w=None, ax=None):
    if t.ndim != 1 or x_true.ndim != 1:
        print("input vectors are not 1D")
        return
    if abs((t[1] - t[0]) - 1) < 1e-10:
        time_label = r"timestep $k$"
    else:
        time_label = r"time $t$ (sec)"
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    # z = np.asarray(z).ravel() # makes the msmt into a 1D vector - suggest
    # changing in the future
    ax.plot(t, x_true, "-.", color=MCcolors["mag"], label="true state")
    if z is not None:
        ax.plot(t, z, ".", color=MCcolors["green"], markersize=2, label="measurement")
    if x_no_w is not None:
        ax.plot(t, x_no_w, "-", color=MCcolors["blue"],label="state w/ no noise")
    ax.grid(True)
    ax.set_xlabel(time_label)
    ax.set_ylabel("state") 
    ax.legend(loc="best")
