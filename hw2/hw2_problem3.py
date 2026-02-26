import numpy as np

p_z2x = np.array([0.1,0.6,0.3])

k = 1
found = False
new_dist = p_z2x

while not found:
    new_dist = np.power(p_z2x, k)
    new_dist /= np.sum(new_dist)
    if new_dist[1] > 0.99:
        found = True
        break
    print(f"K not sufficient: {k}")
    k += 1

print(k)
print(new_dist)

