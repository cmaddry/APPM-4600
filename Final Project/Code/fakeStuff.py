import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from functions import Bowl, Rosenbrock, Himmelblau


## 3D Plots
# Grid for plotting the function
x = np.linspace(-20, 20, 1000)
y = np.linspace(-20, 20, 1000)
X, Y = np.meshgrid(x, y)

evalPoints = [X, Y]

# syntax for 3-D projection
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, np.log(Bowl(evalPoints)), cmap=cm.RdPu_r,
                       linewidth=10, antialiased=False)
plt.show()

# Grid for plotting the function
x = np.linspace(-5, 10, 1000)
y = np.linspace(-25, 25, 1000)
X, Y = np.meshgrid(x, y)

evalPoints = [X, Y]

# syntax for 3-D projection
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, np.log(Rosenbrock(evalPoints)), cmap=cm.RdPu_r,
                       linewidth=10, antialiased=False)
plt.show()


# Grid for plotting the function
x = np.linspace(-10, 10, 1000)
y = np.linspace(-25, 25, 1000)
X, Y = np.meshgrid(x, y)

# syntax for 3-D projection
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, np.log(Himmelblau(evalPoints)), cmap=cm.RdPu_r,
                       linewidth=10, antialiased=False)
plt.show()