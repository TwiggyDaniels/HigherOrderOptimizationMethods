import math
import cmath
import random

import numpy as np
import matplotlib.pyplot as plt

# configure to catch overflows as errors
import warnings
warnings.filterwarnings("error")

# define the golden ratio constant
GOLDEN_RATIO = (math.sqrt(5) + 1) / 2

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
def inexact_line_search(grad, obj_val, x, alpha, beta):
    step_size = 1.0
    while ( eval_objective(x - (step_size * grad))  > 
        ( obj_val - (alpha * step_size * (np.linalg.norm(grad, ord=2)**2)) )):
        step_size *= beta
    return step_size

# Perform the golden search with initial brackets {0, bracket_high} for
# some number of iterations since unimodality guaranteed be claimed on
# the interval provided.
def exact_line_search(x, grad, bracket_high=1, attempts=100, early_stop=1E-5):

    high = bracket_high
    low = 0

    high_minus = high - ( (high - low) / GOLDEN_RATIO)
    low_plus = low + ( (high - low) / GOLDEN_RATIO)

    i = 1
    while i < attempts and abs(high_minus - low_plus) > early_stop:
        if eval_objective(x - (high_minus * grad)) < \
            eval_objective(x - (low_plus * grad)):
            high = low_plus
        else:
            low = high_minus

        high_minus = high - ( (high - low) / GOLDEN_RATIO)
        low_plus = low + ( (high - low) / GOLDEN_RATIO)
    
        i += 1

    return (high + low) / 2

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
    # array to store the objective values at each x_i
    results = np.zeros((iterations,1))
    total_runtime = 0.0

    # declare x_{i-1} for scope reasons
    x_imo = 0

    # get the starting point objective value
    results[0] = eval_objective(arguments[0])

    for i in range(1, iterations):

        # pass the unpacked arguments and hand to the passed function
        arguments, iteration_time = method_iterator(*arguments) 

        # get value at x_i (x_i always first argument)
        x_i = arguments[0]
        results[i] = eval_objective(x_i)
        total_runtime += iteration_time

        # check for early stopping by relative error
        if results[i] == 0 or ( (i > 0) and 
                np.linalg.norm(x_i - x_imo, ord=2) < 
                ( early_stop * np.linalg.norm(x_i, ord=2) ) ):
            return results, i + 1, total_runtime

        # update x_{i-1} for stopping condition
        x_imo = x_i

    return results, iterations, total_runtime

def plot_results(results, title):
    # plot for each method in the data by the key and
    # note that the 0'th element is the array of obj vals

    # plot the objective value graphs
    for i in range(2):
        #plt.subplot(3, 1, i+1)
        for key in results:
            # get the number of iterations the algorithm needed to finish
            complete_iter = results[key][1]
            # create indicies for each iteration
            iterations = np.arange(complete_iter) + 1
    
            # plot only up to the point where it finds the minimum
            plt.plot(iterations, results[key][0][:complete_iter], label=key)
        
        # set up the rest of the line graph
        plt.ylabel('Objective Value')
        plt.xlabel('Iteration')
        plt.title("Linear Comparison")
        plt.grid(True)
        plt.legend()

        # change to log scale
        if i == 1:
            plt.yscale("log")
            plt.yscale("log")

        plt.show()

    m = []
    t = []
    s = []

    for key in results:
        m.append(key)
        s.append(results[key][1])
        # get seconds and convert to milliseconds
        t.append(results[key][2] * 1000)

    # convert t and s to numpy arrays for easier element-wise division
    t = np.array(t)
    s = np.array(s)

    m_idx = np.arange(len(m))

    # plot the time graphs
    for i in range(2):
        #plt.subplot(3, i+1, 3)

        data = t

        # average time plot
        if i == 1:
            data = data/s
            plt.title("Avg Iteration Time")
            plt.bar(m_idx, data)
        # total time per method
        else:
            plt.title("Time Until Stop")
            plt.bar(m_idx, data)
        plt.xticks(m_idx, m)
    
        for y in range(len(m_idx)):
            plt.text(x=m_idx[y]-0.2, y=data[y]+0.02, s="{:.3f}".format(data[y]), size=10)
    
        #plt.subplots_adjust(top=2)
    
        # set up the rest of the line graph
        plt.ylabel("Time (ms)")
        plt.xlabel("Method")
        plt.grid(True)
        # show it
        plt.show()
