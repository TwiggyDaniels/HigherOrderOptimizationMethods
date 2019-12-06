import numpy as np

# Create an x of length n [-1.2, 1.0, -1.2, 1.0, ...]
#
#   n   : number of elements for the vector
#
#   returns:
#       a numpy array (vector) of length n
#
def init_x(n):
    x = np.ones((n))
    x[::2] -= 2.2
    return x

# Evaluate the objective function at x_k
#
#   x   : the particular vector at x_k to evaluate
#
#   returns:
#       val : the value of the objective function, f(x_k)
#
def eval_objective(x):
    val = 0
    for i in range(1, x.shape[0]):
        val += 100 * ((x[i] - ((x[i-1])**2))**2) + ((1 - x[i-1])**2)
    return val

# Find the gradient at some particular x_k
#
#   x   : the particular vector x_k to examine
#
#   returns:
#       grad    : the gradient at x_k
#
def gradient(x):
    grad = np.zeros_like(x)

    for i in range(1, x.shape[0]):
        grad[i] += d_x(x[i], x[i-1])
        grad[i-1] += d_ximo(x[i], x[i-1])

    return grad

# Derivative of the objective function with respect to x_i
#
#   x_i     : the current element of vector x_k
#   x_{i-1} : the previous element of vector x_k
#
#   returns:
#       der_xi    : the derivative
#
def d_x(x_i, x_imo):
    #TODO

# Derivative of the objective function with respect to x_{i-1}
#
#   x_i     : the current element of vector x_k
#   x_{i-1} : the previous element of vector x_k
#
#   returns:
#       der_ximo    : the derivative
#
def d_ximo(x_i, x_imo):
    #TODO

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
    b = (a**2) - RECIPROCAL_KAPPA

    # a ommited since a = 1 for quadratic equation
    discriminant = (b**2) - (4 * 1 * c)

    # find the two quadratic solutions
    soln1 = ((-b - cmath.sqrt(d)) / 2).real
    soln2 = ((-b + cmath.sqrt(d)) / 2).real

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
        ( obj_val - (alpha * s * (np.linalg.norm(grad, ord=2)**2)) )):
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
def iterator(method_iterator, arguments, iterations, early_stop)

    results = np.zero((iterations), dtype=np.int32)
    total_runtime = 0.0

    for i in range(iterations):

        # pass the unpacked arguments and hand to the passed function
        arguments, iteration_time = method_iterator(*arguments) 

        # get value at x_i (x_i always first argument)
        results[i] = eval_objective(arguments[0])
        total_runtime += iteration_time

        # check for early stopping
        if results[i] <= early_stop:
            return results, i + 1, total_runtime, (total_runtime / i)


    return results, iterations, total_runtime, (total_runtime / iterations)
