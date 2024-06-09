import numpy as np
import matplotlib.pyplot as plt


L = 1.0         
beta = 1.0       
p_bar_3D = 1.0   
p_inf = 0.5     
gamma = 1.0      
f_1D = lambda s: 0.0  


N = 100         
ds = L / (N - 1) 


s = np.linspace(0, L, N)


p_1D = np.zeros(N)
A = np.zeros((N, N))


for i in range(1, N - 1):
    A[i, i - 1] = 1
    A[i, i] = -2 - beta * ds**2
    A[i, i + 1] = 1


A[0, 0] = -2 - beta * ds**2 - 2 * gamma * ds
A[0, 1] = 2
A[-1, -2] = 2
A[-1, -1] = -2 - beta * ds**2 - 2 * gamma * ds


b = np.zeros(N)
for i in range(1, N - 1):
    b[i] = -beta * ds**2 * p_bar_3D - ds**2 * f_1D(s[i])
b[0] = -beta * ds**2 * p_bar_3D - 2 * gamma * ds * p_inf - ds**2 * f_1D(s[0])
b[-1] = -beta * ds**2 * p_bar_3D - 2 * gamma * ds * p_inf - ds**2 * f_1D(s[-1])


p_1D = np.linalg.solve(A, b)


plt.figure()
plt.plot(s, p_1D, marker='o', linestyle='-', label='Numerical Solution')
plt.xlabel('s')
plt.ylabel('p_1D')
plt.title('1D Pressure Distribution')
plt.grid(True)
plt.legend()
plt.show()