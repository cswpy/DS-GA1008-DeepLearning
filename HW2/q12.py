import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def calculate_psi(x,y):
    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
        up = abs(y-0.5)
        down = abs(y+0.5)
        right = abs(x-0.5)
        left = abs(x+0.5)
        return min(up, down, right, left)
    else:
        return 0

def g(x,y):
    if x+y <= 0:
        return -1
    else:
        return 1

d = 2
x_1 = []
x_2 = []
f_x = []
x_range = np.arange(-1, 1, 0.01)

for x1 in x_range:
    for x2 in x_range:
        list_val = [-1, 1]
        y = 0
        for z1 in list_val:
            for z2 in list_val:
                y += g(z1, z2) * calculate_psi(x1-z1/2, x2-z2/2)
        x_1.append(x1)
        x_2.append(x2)
        f_x.append(y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_1, x_2, f_x, c=f_x, cmap='summer')
plt.show()