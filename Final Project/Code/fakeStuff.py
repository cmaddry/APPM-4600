import numpy as np
from numpy.linalg import norm

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


def GradientDescentnD(x0, F, J, tol, Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F,tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approximate root, ier = error message, its = num its'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0
        ier = 0
        its = 0
        return


    for its in range(Nmax):
        # Evaluate J and compute its inverse     
        Jeval = J(x0)
        Jtranspose= np.transpose(Jeval)          
    
        # Evaluate F
        Feval = F(x0)

        # Find the step length
        alpha = 0.8 # THIS IS SOMETHING WE WANT TO FIND AN IMPLEMENTATION TO

        p0 = Jtranspose.dot(Feval)

        # Calculate the step 
        x1 = x0 - alpha * p0
    
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1
    ier = 1
    return[xstar, ier, its]