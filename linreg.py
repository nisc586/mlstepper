import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

x_old = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]).reshape(-1, 1)
x = np.hstack([np.ones((11, 1)), x_old])
y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]).reshape(-1, 1)

# linear model:
# y_i = xT_i * beta + eps_i

# solve ordinary least squares
coeffs = linalg.inv(x.T@x) @ x.T @ y
estimates = x @ coeffs

# 
rss = np.sum((estimates - y.mean())**2)  # "explained" variance
tss = np.sum((y - y.mean())**2)  # "total" variance
r_square = 1 - rss/tss  # coefficient of determination

print(r_square)
fig, ax = plt.subplots()
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
plt.scatter(x_old, y)
plt.plot(x_old, estimates, color="red")

plt.show()