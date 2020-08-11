


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#initial conditions
initial_cond = (1, 1, 1)

# Maximum time point and total number of time points
tmax, n = 100, 10000

"""The Lorenz equations."""
def Lorenz(t, X):
    x, y, z = X
    dx = -10 * (x - y)
    dy = 28 * x - y - x * z
    dz = -(8/3) * z + x * y
    return dx, dy, dz


# Integrate the Lorenz equations and evaluating the solution at the time grid t.
#Please note that solve_ivp method uses explicit RK-4 as the default solution method
t = np.linspace(0, tmax, n)
sol = solve_ivp(Lorenz, (0,tmax), initial_cond, t_eval = t)

# Plot the Lorenz attractor using a Matplotlib 3D projection.
fig=plt.figure()
ax = Axes3D(fig)
ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], 'b-', lw=0.5)
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)
plt.tick_params(labelsize=15)
ax.set_title('Lorenz Attractor', fontsize=15)
plt.show()