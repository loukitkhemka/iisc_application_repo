

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

n_samples = 10000

mean = np.array([1,1,1])
cov_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])

initial_points = np.random.multivariate_normal(mean, cov_mat, n_samples)

"""The Lorenz equations."""
def Lorenz(t, X):
    x, y, z = X
    dx = -10 * (x - y)
    dy = 28 * x - y - x * z
    dz = -(8/3) * z + x * y
    return dx, dy, dz
# Maximum time point and total number of time points
tmax, n = 100, 25
solution = np.zeros((n, 3, n_samples))
#initial conditions
initial_cond = (1, 1, 1)

for i in range(0,n_samples):
    initial_cond = initial_points[i,:]
    
    # Integrate the Lorenz equations and evaluating the solution at the time grid t.
    #Please note that solve_ivp method uses explicit RK-4 as the default solution method
    t = np.linspace(0, tmax, n)
    sol = solve_ivp(Lorenz, (0,tmax), initial_cond, t_eval = t)
    solution[:,:, i] = sol.y.T

tstep = 20
#Plotting
plt.figure(1)
h = plt.hist2d(solution[tstep,0,:],solution[tstep,1,:], bins = 36, density = True)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(h[3])
plt.show()

plt.figure(2)
plt.hist2d(solution[tstep,1,:],solution[tstep,2,:], bins = 36, density = True)
plt.xlabel('y')
plt.ylabel('z')
plt.colorbar(h[3])
plt.show()

plt.figure(3)
plt.hist2d(solution[tstep,0,:],solution[tstep,2,:], bins = 36, density = True)
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar(h[3])
plt.show()