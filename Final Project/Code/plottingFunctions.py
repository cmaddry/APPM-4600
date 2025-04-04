import matplotlib.pyplot as plt
import numpy as np
from method_functions import GradientDescent, NewtonDescent
from functions import Himmelblau, HimmelblauJ, HimmelblauH


def PlotHimmelblau(x0 = [-5, 20]):
    # Initial conditions and other parameters
    tol = 1e-16; Nmax = 100000;

    # Finds the roots with Gradient Descent and then returns and prints the output
    GDRoot, error, iterations, GDSteps = GradientDescent(x0, Himmelblau, HimmelblauJ, tol, Nmax)
    print("Root using Gradient Descent:", GDRoot, "\n\t Number of iterations:", iterations)

    # Finds the roots with Newton Descent and then returns and prints the output
    NDRoot, error, iterations, NDSteps = NewtonDescent(x0, Himmelblau, HimmelblauJ, HimmelblauH, tol, Nmax)
    print("Root using Newton Descent:", NDRoot, "\n\t Number of iterations:", iterations)

    # Converts the lists into numpy arrays for plotting
    GDSteps = np.array(GDSteps)
    NDSteps = np.array(NDSteps)

    # Plot of the Rosenbrock function 
    plt.plot(GDSteps[:,0], GDSteps[:,1],
                color='#c61aff',
                marker="o",
                zorder=1,
                label="Gradient Descent")
    plt.plot(NDSteps[:,0], NDSteps[:,1],
                color='#008fb3',
                marker="o",
                zorder=1,
                label="Newton Descent")

    return