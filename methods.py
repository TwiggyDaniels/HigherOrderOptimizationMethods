import time

import numpy as np

from helpers import gradient, find_a_next, find_dynamic_momentum, 
    line_search, iterator

# Inexact Line Search
# Backtracking Constants
ALPHA = 0.5
BETA = 0.5

# estimate of the Lipschitz Constant
LIPSCHITZ = 0.5

# momentum value
MOMENTUM = 0.5

# estimate of the Condition Number
KAPPA = 2
RECIPROCAL_KAPPA = 1 / KAPPA

iterations = 100
input_size = 10

# Perform the gradient descent method
def heavy_ball(x_init, iterations, early_stop=0, alpha, beta):

    # packed arguments for the iterator
    arguments = [x, alpha, beta]

    # call the iterator
    return iterator(gradient_descent_iteration, 
            arguments, iterations, early_stop)

def gradient_descent_iterator(x, alpha, beta):
    
    # get the start time of the iteration
    start_time = time.time()

    # evaluate at x
    grad = gradient(x)
    obj_val = eval_objective(x)

    # perform backtracking line search
    step_size = line_search(grad, obj_val, x, alpha, beta)

    # perform actual gradient descent
    x_next = x - np.dot(step_size, grad)

    return [x_next, alpha, beta], (time.time() - start_time)




# Perform the heavy ball method
def heavy_ball(x_init, iterations, early_stop=0, alpha, beta, momentum):

    # packed arguments for the iterator
    arguments = [x_init, x_init, alpha, beta, momentum]

    # call the iterator
    return iterator(heavy_ball_iteration, arguments, iterations, early_stop)

# Perform a single iteration of the Heavy Ball method
# at a particular x_k and x_{k-1}
def heavy_ball_iteration(x, x_prev, alpha, beta, momentum):
    
    # get the start time of the iteration
    start_time = time.time()

    # evaluate at x
    grad = gradient(x)
    obj_val = eval_objective(x)

    # perform backtracking line search
    step_size = line_search(grad, obj_val, x, alpha, beta)

    # subtract the gradient and add momentum
    x_next = x - np.dot(step_size, grad) + np.dot(momentum, (x - x_prev))

    return [x_next, x, alpha, beta, momentum], (time.time() - start_time)




def p_r_conjugate_gradient():
    
    # get the start time of the iteration
    start_time = time.time()
    #TODO



def acc_grad_descent_iteration(x_init, iterations, early_stop=0, a, lipschitz, recip_kappa):

    # packed arguments for the iterator
    arguments = [x_init, x_init, a, lipschitz, recip_kappa]

    # call the iterator
    return iterator(acc_grad_descent_iteration, arguments, iterations, early_stop)

def acc_grad_descent_iteration(x, y, a, lipschitz, recip_kappa):
    
    # get the start time of the iteration
    start_time = time.time()

    # find x_{k+1} (regular gradient step)
    x_next = y - ((1/lipschitz) * gradient(y))

    # find a_{k+1} from a_k
    a_next = find_a_next(a, recip_kappa)

    # find B_k
    dynamic_momentum = find_dynamic_momentum(a, a_next)

    # find y_{k_1} (sliding step)
    y_next = x_next + dynamic_momentum(x_next - x)

    return [x_next, y_next, a_next, lipschitz, recip_kappa], (start_time - time.time())



# TODO
def fista(x_init, iterations, early_stop=0):

    # packed arguments for the iterator
    arguments = []

    # call the iterator
    return iterator(fista_iteration, arguments, iterations, early_stop)

def fista_iteration():
    
    # get the start time of the iteration
    start_time = time.time()
    #TODO
    pass



def barzilai_borwein_iter(x_init, iterations, early_stop=0):

    # packed arguments for the iterator
    arguments = [x_init, WHAT GOES HERE?>?? FOR X_-1???, gradient(x_init)]

    # call the iterator
    return iterator(acc_grad_descent_iteration, arguments, iterations, early_stop)

def bb_iter(x, x_prev, grad_prev):
    
    # get the start time of the iteration
    start_time = time.time()

    grad = gradient(x)
    r = x - x_prev
    q = grad - grad_prev

    # TODO: confirm you don't need transposes here...
    step_size = np.dot(r, q) / np.dot(q, q)

    x_next = x - (step_size * grad)

    return [x_next, x, grad], (start_time - time.time())
