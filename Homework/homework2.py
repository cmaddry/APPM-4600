import math
import numpy as np
import matplotlib.pyplot as plt

## Question 4 (a)
t = np.linspace(0, np.pi, 31)
y = np.cos(t)
print('the sum is: ', sum(t*y))


## Question 4 (b)
def x(theta, R, delta_r, f, p):
    return R*(1 + delta_r * np.sin(f * theta + p))*np.cos(theta)

def y(theta, R, delta_r, f, p):
    return R*(1 + delta_r * np.sin(f * theta + p))*np.sin(theta)

# values for the first plot
R = 1.2
delta_r = 0.1
f = 15
p = 0
theta = np.linspace(0, 2* np.pi, 100)

# plotting 
fig1 = plt.plot(x(theta, R, delta_r, f, p), y(theta, R, delta_r, f, p))
plt.xlabel(r'x($ \theta $)')
plt.ylabel(r'y($ \theta $)')
plt.title('Wavy circles')
plt.xticks(range(-10, 11))
plt.yticks(range(-10, 11))

# initialization for the for loop
i = 10
delta_r = 0.05
plt.figure()

for i in range(i):
    # values that change with the loop iteration
    R = i
    f = 2 + i
    p = np.random.uniform(0,2,1)[0]

    # plotting
    plt.plot(x(theta, R, delta_r, f, p), y(theta, R, delta_r, f, p))
    plt.xlabel(r'x($ \theta $)')
    plt.ylabel(r'y($ \theta $)')
    plt.title('Wavy circles')
