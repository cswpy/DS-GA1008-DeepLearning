import torch, numpy as np
import matplotlib.pyplot as plt

d_range = [10, 100, 1000, 10000]
figure, axis = plt.subplots(4, figsize=(9,5))

for i in range(len(d_range)):
    d = d_range[i]
    dist = []
    for _ in range(1000):
        x = np.random.normal(loc=0, scale=1/np.sqrt(d), size=d)
        norm_2_squared = np.sum(np.square(x))
        dist.append(norm_2_squared)
    dist.sort()
    mean = np.mean(dist)
    std = np.std(dist)
    unique, counts = np.unique(dist, return_counts=True)
    #print(unique, counts)
    axis[i].plot(unique, counts)
    axis[i].set_title(f"d={d}, mean={mean:.3f}, std={std:.3f}")
    #axis[i].set_le
figure.tight_layout()
plt.show()
