import matplotlib.pyplot as plt
import numpy as np


# ## vectors 
# x = [1, 2, 3]
# y = np.array([1,2,3])
# print('this is 3y', 3*y)


# ## Plotting
# # Now I can make some vectors to plot
# X = np.linspace(0, 2 * np.pi, 100)
# Ya = np.sin(X)
# Yb = np.cos(X)
# plt.plot(X, Ya)
# plt.plot(X, Yb)
# plt.show()

# #What size is X? Can you explain how the command linspace works?
# #If we want to add labels, we can
# X = np.linspace(0, 2 * np.pi, 100)
# Ya = np.sin(X)
# Yb = np.cos(X)
# plt.plot(X, Ya)
# plt.plot(X, Yb)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


## 3.2 Excercises: The Basics
x = np.linspace(1, 10, 10)
y = np.arange(1, 11, 1)


print('the first three entries of x are', x[0:3]) 


w = 10**(-np.linspace(1,10,10))
# these are the entires of w: [1.e-01 1.e-02 1.e-03 1.e-04 1.e-05 1.e-06 1.e-07 1.e-08 1.e-09 1.e-10]
x = np.arange(1, len(w) + 1, 1) 
print(x)

a = plt.semilogy(x, w)
plt.xlabel('x')
plt.ylabel('w')
s = 3*w
plt.semilogy(x,s)
plt.savefig('test.png')



## Practical Code Design
import numpy as np
import numpy.linalg as la
import math
def driver():
    n = 2
    x = np.linspace(0,1,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    
    y = [0,1]
    w = [1,0]
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    # print the output
    print('the dot product is : ', dp)
    return

def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp
driver()

# 1. I changed the vectors to the unit vectors

# 2. Matrix multiplication code ONLY SQUARE MATRICIES
def driver():
    n = 2
    x = np.linspace(0,1,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    
    y = [0,1]
    w = [1,0]
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    # print the output
    print('the dot product is : ', dp)
    return

def matrixMulti(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        for i in range(n):
            dp = dp + x[i, j]*y[i, j]
    return dp
driver()