"""
Example calculating and plotting a 2D covariance ellipse

MAE 6760 Model Based Estimation
Cornell University
M Campbell
"""

import numpy as np
import matplotlib.pyplot as plt

from calculate_ellipse import calculate_ellipse

# Example inputs (same as MATLAB example)
X = np.array([0.0, 0.0])
P = np.array([[6.5, 2.5],
              [2.5, 6.5]])

# Compute ellipse
Xe, Ye, U, S, th = calculate_ellipse(X, P, nsig=1, np_points=50)

print("Rotation angle (deg):", th)
print("U:\n", U)
print("S:\n", S)

# Plot
plt.figure()
plt.plot(Xe, Ye, 'b-', label='1-sigma ellipse')
plt.plot(X[0], X[1], 'ro', label='Mean')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Covariance Ellipse')
plt.show()