import numpy as np
from numpy.linalg import inv 
from numpy.linalg import norm 

def GradientDescent(x0, F, J, tol, Nmax):
    ''' Gradient descent: Implementation of gradient descent for optimization'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, tol = tolerance, Nmax = max number of iterations'''
    ''' Outputs: xstar= approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return[xstar, ier, its, x0]
    
    xStep = [x0]
    for its in range(Nmax):
        # Evaluate J and compute its inverse     
        Jeval = J(x0)

        # Find the step length
        dk = -Jeval
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=False)

        # Calculate the step 
        x1 = x0 - alpha * Jeval
        xStep = xStep + [x1.tolist()]
    
        # If we found the root (to within the tolerance), return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def NewtonDescent(x0, F, J, H, tol, Nmax):

    ''' NewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, xStep]

    xStep = [x0]

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)
        
        # Evaluates the Hessian
        Heval = H(x0)      

        # Find the step length wihth LU solve instead of inverse
        p0 = np.linalg.solve(Heval, Jeval)
        dk = -p0

        # Finds alpha with a back tracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=False)

        # Calculate the step 
        x1 = x0 - alpha*p0
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def LazyNewtonDescent(x0, F, J, H, tol, Nmax):

    ''' NewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, xStep]

    xStep = [x0]

    # Evaluates the Hessian and computes its inverse
    Heval = H(x0)   

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = np.linalg.solve(Heval, Jeval)
        dk = -p0

        # Finds alpha with a backtracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk)

        # Calculate the step 
        x1 = x0 - alpha*p0
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def BFGSNewtonDescent2(x0, F, J, H, tol, Nmax):

    ''' BFGSNewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, [x0]]

    xStep = [x0]

    # Initial Guess is the identity matrix
    B0 = np.eye(2,2)

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = np.linalg.solve(B0, Jeval)
        dk = -p0

        # Finds alpha with a backtracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=True)

        # Calculate the step 
        x1 = x0 + dk*alpha
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]

        # BFGS update 
        s = x1 - x0
        y = J(x1) - J(x0)       
        B0 = BFGS2(s, y, B0)
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def BFGS2(s, y, B0):
    # Initialization of 
    B1 = np.zeros([2,2])

    # Terms used in the calculation
    term1 = (B0 @ np.outer(s,s) @ B0)/(s.T @ B0 @ s)
    term2 = np.outer(y, y)/np.dot(y, s)

    # Update for the Hessian approximate
    B1 = B0 - term1 + term2

    return B1

def BFGSNewtonDescent(x0, F, J, H, tol, Nmax):

    ''' BFGSNewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, [x0]]

    xStep = [x0]

    # Initial Guess is the identity matrix
    B0inv = np.eye(2,2)
    # B0inv = inv(H(x0))

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = B0inv.dot(Jeval)
        dk = -p0

        # Finds alpha with a backtracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=True)

        # Calculate the step 
        x1 = x0 + dk*alpha
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(x1 - x0) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]

        # BFGS update 
        s = x1 - x0
        y = J(x1) - J(x0)       
        B0inv = BFGS(s, y, B0inv)
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def BFGS(s, y, B0_inv):
    # Identity 
    I = np.eye(2,2)

    # Rho
    rho = 1.0 / np.dot(y, s)

    # Terms for the Hessian Approximation
    term1 = I - np.dot(rho, np.outer(s,y))
    term2 = I - np.dot(rho, np.outer(y,s))
    term3 = np.dot(rho, np.outer(s,s))

    # Calculation of the inverse Hessian approximation
    B1_inv = term1 @ B0_inv @ term2 + term3

    return B1_inv

def DFPNewtonDescent(x0, F, J, H, tol, Nmax):

    ''' BFGSNewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, [x0]]

    xStep = [x0]

    # Initial Guess is the identity matrix
    # B0inv = np.eye(2,2)
    B0inv = inv(H(x0))

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = B0inv.dot(Jeval)
        dk = -p0

        # Finds alpha with a backtracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=True, p=0.001)

        # Calculate the step 
        x1 = x0 + dk*alpha
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(F(x1)) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]

        # BFGS update 
        s = x1 - x0
        y = J(x1) - J(x0)       
        B0inv = DFP(s, y, B0inv)
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def DFP(s, y, B0_inv):

    if abs((y.T @ B0_inv @ y)) < 1e-8 or abs(np.dot(y,s)) < 1e-8:
        return B0_inv

    # Terms for the Hessian Approximation
    term1 = (B0_inv @ np.outer(y,y) @ B0_inv)/(y.T @ B0_inv @ y)
    term2 = np.outer(s,s)/np.dot(y,s)

    # Calculation of the inverse Hessian approximation
    B1_inv = B0_inv - term1 + term2

    return B1_inv

def DFPNewtonDescent2(x0, F, J, H, tol, Nmax):

    ''' BFGSNewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, [x0]]

    xStep = [x0]

    # Initial Guess is the identity matrix
    # B0 = np.eye(2,2)
    B0 = H(x0)
    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)    

        # Find the step length
        p0 = np.linalg.solve(B0, Jeval)
        dk = -p0

        # Finds alpha with a backtracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=True)

        # Calculate the step 
        x1 = x0 + dk*alpha
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(F(x1))  < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]

        # BFGS update 
        s = x1 - x0
        y = J(x1) - J(x0)       
        B0 = DFP2(s, y, B0)
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def DFP2(s, y, B0):
    # Identity 
    I = np.eye(2,2)

    # Rho
    rho = 1.0 / np.dot(y, s)

    # Terms for the Hessian Approximation
    term1 = I - np.dot(rho, np.outer(y,s))
    term2 = I - np.dot(rho, np.outer(s,y))
    term3 = np.dot(rho, np.outer(y,y))

    # Calculation of the inverse Hessian approximation
    B1 = term1 @ B0 @ term2 + term3

    return B1

def SR1NewtonDescent(x0, F, J, H0, tol, Nmax):

    ''' BFGSNewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, xStep]

    xStep = [x0]

    # Inital B0 is the hessian at x0
    B0 = H0

    for its in range(Nmax):
        # Evaluate J    
        Jeval = J(x0)

        # Find the step length
        p0 = np.linalg.solve(B0, Jeval)
        dk = -p0

        # Update alpha with a backtracking line search
        alpha = backTrackingLineSearch(x0, F, J, dk, wolfe=True)

        # Calculate the step 
        x1 = x0 + alpha*dk
        xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(F(x1)) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]
        
        # Stuff for BFGS
        s = x1 - x0
        y = J(x1) - J(x0)

        if abs(s.T @ (y - B0 @ s)) < 1e-8 * norm(s) * norm(y - B0 @ s):
            # BFGS update        
            B0 = SR1(s, y, B0)

        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

def SR1(s, y, B0):

    # Initialize B1
    B1 = np.zeros([2,2])

    # Calculate the numerator and denominator in the SR1 update
    num = (y - B0 @ s) @ (y - B0 @ s).T
    denom = (y - B0 @ s).T @ s

    if abs(denom) < 1e-8:
        return B0 

    # Calculate the SR1 update
    B1 = B0 + num/denom

    return B1

def backTrackingLineSearch(xk, F, J, dk, wolfe=False, maxIts = 100, p = 0.5, c1=1e-4, c2=0.9):

    ''' backTrackingLineSearch: calculate optimal alpha using Armijo condition'''
    ''' inputs: xk = step location, F = function, grad_xk = gradient of F, dk = descent direction, p = step length, c = Armijo constant'''
    ''' Outputs: alpha = step size'''

    # Initial guess for alpha
    alpha = 1.0
    its = 0

    if wolfe == False:
        # While the Amrijo condition is not satisfied, keep decreasing alpha
        while F(xk + alpha * dk) > F(xk) + c1 * alpha * np.dot(J(xk), dk) and its < maxIts: 
            
            # Update alpha
            alpha = p * alpha
            its += 1

    elif wolfe == True:
        # While the Amrijo and Wolfe condition is not satisfied, keep decreasing alpha
        while F(xk + alpha * dk) > F(xk) + c1 * alpha * np.dot(J(xk), dk) or abs(np.dot(J(xk + alpha * dk), dk)) > c2 * abs(np.dot(J(xk), dk)) and its < maxIts: 
            if alpha < 1e-8:
                break
            # Update alpha
            alpha = p * alpha
            its += its

    # Return alpha
    return alpha



def trustRegionMethod(x0, F, J, H, tol, Nmax, deltak=1, eta=0.5):

    ''' NewtonDescent: use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, F = Function, J = Jacobian of F, H = Hessian of F, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar = approximate root, ier = error message, its = num its, xStep = xk for each step k'''

    # Checks to see if the function is already at the root
    if norm(F(x0)) == 0: 
        xstar = x0; ier = 0; its = 0
        return [xstar, ier, its, xStep]

    xStep = [x0]

    for its in range(Nmax):
        # Evaluate teh function
        Feval = F(x0)

        # Evaluate J    
        Jeval = J(x0)

        # Evaluates the Hessian
        Heval = H(x0)      

        # Updates the step
        pk = solveTRMin(Jeval, Heval, deltak)

        # Calculates rho
        pRed = -Jeval.T @ pk - 0.5 * pk.T @ Heval @ pk
        aRed = Feval - F(x0 + pk)
        rhok = aRed / pRed

        # Update trust region radius
        if rhok < 0.25:
            deltak *= 0.25
        elif rhok > 0.75 and np.linalg.norm(pk) == deltak:
            deltak = min(2 * deltak, 10.0)

        # Accept or reject the step
        if rhok > eta:
            x1 = x0 + pk  # Accept the step
            xStep = xStep + [x1.tolist()]
       
        # If we found the root (to within the tolerance), 
        # return it and 0 for the error message
        if (norm(J(x1)) < tol):
           xstar = x1; ier = 0
           return[xstar, ier, its, xStep]
        
        x0 = x1
    
    # If we didn't find the root, return the step and an error of 1
    xstar = x1; ier = 1
    return[xstar, ier, its, xStep]

from scipy.linalg import solve
from scipy.optimize import minimize_scalar

def solveTRMin(Jeval, Heval, deltak):
    # Define the quadratic model Q(s) = grad.T @ s + 0.5 * s.T @ hess @ s
    def quadratic_model(alpha):
        I = np.eye(len(Jeval))

        s = -solve(Heval + alpha * I, Jeval)
        return np.linalg.norm(s) - deltak

    # Solve for optimal alpha (Lagrange multiplier)
    result = minimize_scalar(lambda alpha: abs(quadratic_model(alpha)), bounds=(0, 1e5), method='bounded')
    
    alpha_opt = result.x
    s = -solve(Heval + alpha_opt * np.eye(len(Jeval)), Jeval)
    return s

    return 1

def mk(xk, F, J, H, pk):
    # Evaluate 
    Feval = F(xk)
    Jeval = J(xk)
    Heval = H(xk) 

    return Feval + Jeval @ pk + 0.5*pk.T @ Heval @ pk