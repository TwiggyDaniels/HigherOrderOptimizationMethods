import cmath
import random

import numpy as np
import matplotlib.pyplot as plt

# Create an x of length n [-1.2, 1.0, -1.2, 1.0, ...]
#
#   n   : number of elements for the vector
#
#   returns:
#       a numpy array (vector) of length n
#
def init_x(n):
    x = np.ones((n), dtype=np.float64)
    x[::2] = -1.2
    return x

# Evaluate the objective function at x_k
#
#   x   : the particular vector at x_k to evaluate
#
#   returns:
#       val : the value of the objective function, f(x_k)
#
def eval_objective(x):
    val = 0.0
    # since the sum starts at the second element...
    for i in range(1, x.shape[0]):
        #val += 100 * ((x[i] - (x[i-1]**2))**2) + ((1 - x[i-1])**2)
        val += ((x[i] - (x[i-1]**2))**2) + ((1 - x[i-1])**2)
    return val

# Find the gradient at some particular x_k
#
#   x   : the particular vector x_k to examine
#
#   returns:
#       grad    : the gradient at x_k
#
def gradient(x):
    grad = np.zeros_like(x, dtype=np.float64)
    # since the sum starts at the second element...
    for i in range(1, x.shape[0]):
        grad[i] += d_x(x[i], x[i-1])
        grad[i-1] += d_ximo(x[i], x[i-1])
    return grad

# Derivative of the objective function with respect to x_i
#
#   x_i     : the current element of vector x_k
#   x_imo   : the previous element of vector x_k
#
#   returns:
#       der_xi    : the derivative
#
def d_x(x_i, x_imo):
    #der_xi = 200 * (x_i - (x_imo**2)) 
    der_xi = 2 * (x_i - (x_imo**2)) 
    return der_xi

# Derivative of the objective function with respect to x_{i-1}
#
#   x_i     : the current element of vector x_k
#   x_imo   : the previous element of vector x_k
#
#   returns:
#       der_ximo    : the derivative
#
def d_ximo(x_i, x_imo):
    #der_ximo = (-400 * x_i * x_imo) + (400 * (x_imo**3)) + (2 * x_imo) - 2
    der_ximo = (-4 * x_i * x_imo) + (4 * (x_imo**3)) + ((2 * x_imo)/100) - (2/100)
    return der_ximo

# Attempts to solve a the quadratic equation to find a_{k+1}. Gives
# a_k + some uniform noise if none of the solutions are in (0, 1)
#
#   a           : the current a_k
#   recip_kappa : the reciprocal of the estimate condition number
#
#   returns:
#       some float value a_{k+1} in (0, 1)
#
def find_a_next(a, recip_kappa):
    c = -(a**2)
    b = (a**2) - recip_kappa

    # a ommited since a = 1 for quadratic equation
    discr = (b**2) - (4 * 1 * c)

    # find the two quadratic solutions
    soln1 = ((-b - cmath.sqrt(discr)) / 2).real
    soln2 = ((-b + cmath.sqrt(discr)) / 2).real

    if (soln1 < 0 or soln1 > 1):
        if (soln2 < 0 or soln2 > 1):
            return max(soln1,  soln2)
        else:
            return soln1
    else:
        if (soln2 < 0 or soln2 > 1):
            return soln2
        # return a_prev with some minor noise if no suitable value
        else:
            a_next = a + random.uniform(-1E-3, 1E-3)
            while (a_next < 0 or a_next > 1):
                a_next = a + random.uniform(-1E-3, 1E-3)
            return a_next

# Find the dynamic momentum, B_k from a_k and a_{k+1}
#
#   a   : the value of a_k
#   a   : the value of a_{k+1}
#
#   returns:
#       the value of the momentum, B_k
#
def find_dynamic_momentum(a, a_next):
    return (a * (1 - a_next)) / ((a**2) + a_next)

# Perform inexact line-search with backtracking
#
#   grad        : gradient, f'(x_k)
#   obj_val     : the value of the objective function, f(x_k)
#   x           : the particular vector at x_k to evaluate
#
#   returns:
#       step_size   : the step size for the iteration update
#
def line_search(grad, obj_val, x, alpha, beta):
    step_size = 1.0
    while ( eval_objective(x - (step_size * grad))  > 
        ( obj_val - (alpha * step_size * (np.linalg.norm(grad, ord=2)**2)) )):
        step_size *= beta
    return step_size

# Optimize the objective function with the provided argument
#
#   method_iterator : a function to perform a single iteration of a method 
#                   such as Gradient Descent or Heavy Ball
#   arguments       : the arguments for the iteration of method_iterator
#   iterations      : maximum number of iterations to perform
#   early_stop      : an early stopping distance for the objective function
#
#   returns:
#       results         : an array of the objectives value at every iteration
#       used_iterations : the total number of iterations performed (early stop)       
#       total_runtime   : total method runtime
#       average_runtime : average runtimer per iteration
#
def iterator(method_iterator, arguments, iterations, early_stop):

    results = np.zeros((iterations,1))
    total_runtime = 0.0

    # get the starting point objective value
    results[0] = eval_objective(arguments[0])

    for i in range(1, iterations):

        # pass the unpacked arguments and hand to the passed function
        arguments, iteration_time = method_iterator(*arguments) 

        # get value at x_i (x_i always first argument)
        results[i] = eval_objective(arguments[0])
        total_runtime += iteration_time

        # check for early stopping if method hasn't
        # changed by some amount since last iteration
        if results[i] == 0 or ( (i > 0) and abs(results[i] - results[i-1]) <= early_stop):
            return results, i + 1, total_runtime

    return results, iterations, total_runtime

def line_graph(results, title):
    # plot for each method in the data by the key and
    # note that the 0'th element is the array of obj vals
    plt.subplot(2, 1, 1)
    for key in results:
        # get the number of iterations the algorithm needed to finish
        complete_iter = results[key][1]
        # create indicies for each iteration
        iterations = np.arange(complete_iter) + 1

        # plot only up to the point where it finds the minimum
        plt.plot(iterations, results[key][0][:complete_iter], label=key)
        plt.ylabel('Objective Value f(x)')
        plt.xlabel('Iteration')
        plt.title('Method Convergence')
        plt.grid(True)
    
    # set up the rest of the line graph
    plt.ylabel('Objective Value f(x)')
    plt.xlabel('Iteration')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    # show it
#    plt.show()

#def time_graph(results, title):

    plt.subplot(2, 1, 2)
    m = []
    t = []

    for key in results:
        m.append(key)
        t.append(results[key][2])

    y_val = np.arange(len(m))
    plt.bar(y_val, t)
    plt.xticks(y_val, m)

    # set up the rest of the line graph
    plt.ylabel('Time in Seconds')
    plt.xlabel('Method')
    plt.title(title)
    plt.grid(True)
    # show it
    plt.show()
