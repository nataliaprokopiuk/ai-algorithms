"""
WSI PROJECT 1

This program implements the gradient descent algorithm
in order to find the minimum and maximum of the given function
"""
import numpy as np

# f(x,y)
def f(vector):
    x = vector[0]
    y = vector[1]
    return (9*x*y)/np.exp(x**2 + 0.5*x + y**2)

# function that calculates partial derivatives
# and returns the f(x,y) function's gradient vector
def gradient(vector):
    x = vector[0]
    y = vector[1]

    df_dx = np.exp(-x**2 - 0.5*x - y**2)*(9*y - 18*x**2*y - 4.5*x*y)
    df_dy = np.exp(-x**2 - 0.5*x - y**2)*(9*x - 18*x*y**2)

    gradient = np.array([df_dx, df_dy])

    return gradient

# function that searches for the function's minimum
def gradient_descent_minimum(start, learn_rate, iterations, min_step=1e-06):
    vector = start

    for i in range(iterations):
        step = -learn_rate * gradient(vector)
        # stop earlier if the steps are small enough
        if np.all(np.abs(step) <= min_step):
            print(i)
            break
        vector += step
    
    return vector

# function that searches for the function's maximum
def gradient_descent_maximum(start, learn_rate, iterations, min_step=1e-06):
    vector = start

    for i in range(iterations):
        step = learn_rate * gradient(vector)
        # stop earlier if the steps are small enough
        if np.all(np.abs(step) <= min_step):
            print(i)
            break
        vector += step
    
    return vector


if __name__ == "__main__":
    # minimum_vector = gradient_descent_minimum(start=[0,0], learn_rate=0.01, iterations=100000)
    # print("minimum (argument): " + str(minimum_vector))
    # print("minimum (value): " + str(f(minimum_vector)))

    print()

    maximum_vector = gradient_descent_maximum(start=[10,10], learn_rate=0.01, iterations=1000)
    print("maximum (argument): " + str(maximum_vector))
    print("maximum (value): " + str(f(maximum_vector)))