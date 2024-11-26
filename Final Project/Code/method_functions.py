import numpy as np
from numpy.linalg import inv 
from numpy.linalg import norm 

def GradientDescent(x0, F, J, tol, Nmax):

    ''' Gradient descent: Implementation of gradient descent for optimization'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, tol = tolerance, Nmax = max number of iterations'''
    ''' Outputs: xstar= approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0
        ier = 0
        its = 0
        return[xstar, ier, its, x0]

    
    xStep = [x0]
    for its in range(Nmax):
        # Evaluate J and compute its inverse     
        Jeval = J(x0)

        # Find the step length
        dk = -Jeval
        alpha = backTrackingLineSearch(x0, F, Jeval, dk)

        # Calculate the step 
        x1 = x0 - alpha * Jeval
        xStep = xStep + [x1.tolist()]

    
        # If we found the root (to within the tolerance), return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1
    ier = 1
    return[xstar, ier, its, xStep]


def NewtonDescent(x0, F, J, H, tol, Nmax):

    ''' NewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0
        ier = 0
        its = 0
        return

    xStep = [x0]

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)
        
        # Evaluates the Hessian and computes its inverse
        Heval = H(x0)   
        Hinv = inv(Heval)      

        # Find the step length
        p0 = Hinv.dot(Jeval)
        dk = -p0

        # alpha = backTrackingLineSearch(x0, F, Jeval, dk, p = 0.5, alpha=1, c=1e-4)
        alpha = backTrackingLineSearch(x0, F, Jeval, dk)

        # Calculate the step 
        x1 = x0 - alpha*p0
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1
    ier = 1
    return[xstar, ier, its, xStep]

def LazyNewtonDescent(x0, F, J, H, tol, Nmax):

    ''' NewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0
        ier = 0
        its = 0
        return

    xStep = [x0]

    # Evaluates the Hessian and computes its inverse
    Heval = H(x0)   
    Hinv = inv(Heval)  

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = Hinv.dot(Jeval)
        dk = -p0

        alpha = backTrackingLineSearch(x0, F, Jeval, dk)

        # Calculate the step 
        x1 = x0 - alpha*p0
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1
    ier = 1
    return[xstar, ier, its, xStep]

def BFGSNewtonDescent(x0, F, J, H, tol, Nmax):

    ''' BFGSNewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0
        ier = 0
        its = 0
        return

    xStep = [x0]

    # Evaluates the Hessian and computes its inverse
    # B0 = H(x0)   
    B0 = np.eye(len(x0))
    B0inv = inv(B0)  

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = B0inv.dot(Jeval)
        dk = -p0

        alpha = backTrackingLineSearch(x0, F, Jeval, dk, p=0.9, c=0.5)

        # Calculate the step 
        x1 = x0 - alpha*p0
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, ier, its, xStep]
        
        # Stuff for BFGS
        s = x1 - x0
        y = J(x1) - J(x0)

        if (y - B0 @ s).T @ s < 1e-8:
            continue
        else:
            # BFGS update        
            B0 = BFGS(s, y, B0)
            B0inv = inv(B0)  
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1
    ier = 1
    return[xstar, ier, its, xStep]

def BFGS(s, y, B0):

    B1 = np.zeros([2,2])

    # num = (y - B0 @ s) @ (y - B0 @ s).T
    a = y - B0 @ s

    num = np.outer(a,a)
    denom = (y - B0 @ s).T @ s

    B1 = B0 + num/denom.item()

    return B1

def backTrackingLineSearch(xk, F, grad_xk, dk, p = 0.5, c=1e-3):

    ''' backTrackingLineSearch: calculate optimal alpha using Armijo condition'''
    ''' inputs: xk = step location, F = function, grad_xk = gradient of F, dk = descent direction, p = step length, c = Armijo constant'''
    ''' Outputs: alpha = step size'''

    # Initial guess for alpha
    alpha = 1.0
    
    # While the Amrijo condition is not satisfied, keep decreasing alpha
    while F(xk + alpha * dk) > F(xk) + c * alpha * np.dot(grad_xk, dk): 
        
        # Update alpha
        alpha = p * alpha

    # Return alpha
    return alpha