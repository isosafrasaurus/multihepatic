# @title Solve 1D PDE by finite difference method
import numpy as np
import matplotlib.pyplot as plt

L = 1.0         # Domain length
beta = 1.0       # Coefficient
p_bar_3D = 1.0   # Mean 3D pressure
p_inf = 0.5     # Far-field pressure
gamma = 1.0      # Boundary condition coefficient
f_1D = lambda s: 0.0  # Source term

N = 100         # Number of grid points
ds = L / (N - 1)

s = np.linspace(0, L, N)

# pressure and coefficient matrix
p_1D = np.zeros(N)
A = np.zeros((N, N))

# Interior points
for i in range(1, N - 1):
    A[i, i - 1] = 1
    A[i, i] = -2 - beta * ds**2
    A[i, i + 1] = 1

# Apply Robin boundary at s = 0
A[0, 0] = gamma * ds - 1
A[0, 1] = 1

# Apply Robin boundary at s = L
A[-1, -1] = 1 - gamma * ds
A[-1, -2] = -1

# right-hand side vector
b = np.zeros(N)
for i in range(1, N - 1):
    b[i] = -beta * ds**2 * p_bar_3D - ds**2 * f_1D(s[i])

# boundary conditions in right-hand side vector
b[0] = gamma * ds * p_inf
b[-1] = -gamma * ds * p_inf

p_1D = np.linalg.solve(A, b)

plt.figure()
plt.plot(s, p_1D, marker='o', linestyle='-')
plt.xlabel('s')
plt.ylabel('p_1D')
plt.title('1D numpy solution')
plt.grid(True)
plt.legend()
plt.show()