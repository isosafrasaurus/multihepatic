import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
L = 1.0         # Domain length
beta = 1.0       # Coefficient
p_bar_3D = 1.0   # Mean 3D pressure
p_inf = 0.5     # Far-field pressure
gamma = 1.0      # Boundary condition coefficient
f_1D = lambda s: 0.0  # Source term (modify as needed)

# Numerical parameters
N = 100         # Number of grid points
ds = L / (N - 1) # Grid spacing

# Create grid
s = np.linspace(0, L, N)

# Initialize pressure and coefficient matrix
p_1D = np.zeros(N)
A = np.zeros((N, N))

# Construct the coefficient matrix A
for i in range(1, N - 1):
    A[i, i - 1] = 1
    A[i, i] = -2 - beta * ds**2
    A[i, i + 1] = 1

# Apply Robin boundary conditions
A[0, 0] = -2 - beta * ds**2 - 2 * gamma * ds
A[0, 1] = 2
A[-1, -2] = 2
A[-1, -1] = -2 - beta * ds**2 - 2 * gamma * ds

# Construct the right-hand side vector
b = np.zeros(N)
for i in range(1, N - 1):
    b[i] = -beta * ds**2 * p_bar_3D - ds**2 * f_1D(s[i])
b[0] = -beta * ds**2 * p_bar_3D - 2 * gamma * ds * p_inf - ds**2 * f_1D(s[0])
b[-1] = -beta * ds**2 * p_bar_3D - 2 * gamma * ds * p_inf - ds**2 * f_1D(s[-1])

# Solve for p_1D
p_1D = np.linalg.solve(A, b)

# Plotting
plt.figure()
plt.plot(s, p_1D, marker='o', linestyle='-', label='Numerical Solution')
plt.xlabel('s')
plt.ylabel('p_1D')
plt.title('1D Pressure Distribution')
plt.grid(True)
plt.legend()
plt.show()