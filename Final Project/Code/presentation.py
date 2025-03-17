import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import norm 

from method_functions import GradientDescent, NewtonDescent
from functions import Bowl, BowlJ, BowlH


# Initial conditions and parameters
x0 = [-18, -18]; tol = 1e-16; Nmax = 100000

# Finds the roots with Gradient Descent and then returns and prints the output
root, error, iterations, GDSteps = GradientDescent(x0, Bowl, BowlJ, tol, Nmax)
print("Root:", root, "\n\t Number of iterations:", iterations)

# # Finds the roots with Newton Descent and then returns and prints the output
# root, error, iterations, NDSteps = NewtonDescent(x0, Bowl, BowlJ, BowlH, tol, Nmax)
# print("Root:", root, "\n\t Number of iterations:", iterations)

# Converts the lists into numpy arrays for plotting
GDSteps = np.array(GDSteps)
# NDSteps = np.array(NDSteps)

# Grid for plotting the function
x = np.linspace(-20, 20, 1000)
y = np.linspace(-20, 20, 1000)
X, Y = np.meshgrid(x, y)
evalPoints = [X, Y]

# Plot of the Rosenbrock function 
plt.plot(GDSteps[:,0], GDSteps[:,1],
            color='#c61aff',
            marker="o",
            label="Gradient Descent",
            zorder=1)
# plt.plot(NDSteps[:,0], NDSteps[:,1],
#             color='#008fb3',
#             marker="o",
#             label="Newton Descent",
#             zorder=1)
plt.contour(X, Y, Bowl(evalPoints),
            levels=25,
            cmap="cool",
            zorder=0)
plt.legend(); plt.title("Quadratic function." + r" $f(x,y) = 6x^2 + y^2$"); plt.xlabel("x"); plt.ylabel("y")
plt.show()
 