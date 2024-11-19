import numpy as np

# Defining the Rosenbrock function and its Jacobian
def Rosenbrock(x):
    ''' a = 1'''
    a = 1; b = 100;
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def RosenbrockJ(x):
    a = 1; b = 100;

    J = np.zeros(2)

    J[0] = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    J[1] = 2*b*(x[1] - x[0]**2)

    return J

def RosenbrockH(x):
    a = 1; b = 100;

    H = np.zeros([2,2])

    H[0,0] = 2 + 8*b*x[0]**2 - 4*b*(x[1] - x[0]**2)
    H[0,1] = -4*b*x[0]
    H[1,0] = -4*b*x[0]
    H[1,1] = 2*b

    return H

def Bowl(x):
    return x[0]**2 + x[1]**2

def BowlJ(x):
    J = np.zeros(2)

    J[0] = 2*x[0]
    J[1] = 2*x[1]
    
    return J

def BowlH(x):
    H = np.zeros([2,2])

    H[0,0] = 2
    H[0,1] = 0
    H[1,0] = 0
    H[1,1] = 2

    return H

def Himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def HimmelblauJ(x):
    J = np.zeros(2)

    J[0] = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    J[1] = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    
    return J

def HimmelblauH(x):
    H = np.zeros([2,2])

    H[0,0] = 2 + 8*x[0]**2 + 4*(x[1] + x[0]**2 - 11)
    H[0,1] = 4*x[0] + 4*x[1]
    H[1,0] = 4*x[0] + 4*x[1]
    H[1,1] = 2 + 8*x[1]**2 + 4*(x[0] + x[1]**2 - 7)

    return H