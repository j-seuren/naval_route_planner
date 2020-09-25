import matplotlib.pyplot as plt
import numpy as np

scaleFactor = .1

n = 100

scale = (n-1) * scaleFactor

N = 1000
k = [np.ceil((np.random.exponential(scale=scale, size=1))).item() for _ in range(N)]

fig, ax = plt.subplots()

ax.hist(k, bins=(n-1)//2)

plt.show()