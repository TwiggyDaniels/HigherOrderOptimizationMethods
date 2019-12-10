import time
import math

import numpy as np

from helpers import eval_objective, gradient, find_a_next, find_dynamic_momentum, \
    inexact_line_search, exact_line_search, iterator, polak_rebiere

# Perform the gradient descent method
def gradient_descent(x_init, iterations, alpha, beta, early_stop=0):

    # packed arguments for the iterator
    arguments = [x_init, alpha, beta]

    # call the iterator
    results, iters, total_runtime = iterator(gd_iter, arguments, iterations, early_stop)
    return results, iters, total_runtime

def gd_iter(x, alpha, beta):
    
    # get the start time of the iteration
    start_time = time.time()

    # evaluate at x
    grad = gradient(x)
    obj_val = eval_objective(x)

    # perform backtracking line search
    step_size = inexact_line_search(grad, obj_val, x, alpha, beta)

    # perform actual gradient descent
    x_next = x - np.dot(step_size, grad)
    
    return [x_next, alpha, beta], (time.time() - start_time)




# Perform the heavy ball method
def heavy_ball(x_init, iterations, alpha, beta, momentum, early_stop=0):

    # packed arguments for the iterator
    arguments = [x_init, x_init, alpha, beta, momentum]

    # call the iterator
    results, iters, total_runtime = iterator(hb_iter, arguments, iterations, early_stop)
    return results, iters, total_runtime 

# Perform a single iteration of the Heavy Ball method
# at a particular x_k and x_{k-1}
def hb_iter(x, x_prev, alpha, beta, momentum):
    
    # get the start time of the iteration
    start_time = time.time()

    # evaluate at x
    grad = gradient(x)
    obj_val = eval_objective(x)

    # perform backtracking line search
    step_size = inexact_line_search(grad, obj_val, x, alpha, beta)

    # subtract the gradient and add momentum
    x_next = (x - np.dot(step_size, grad)) + np.dot(momentum, (x - x_prev))

    return [x_next, x, alpha, beta, momentum], (time.time() - start_time)



def conjugate_gradient(x_init, iterations, bracket_high, epsilon=0, early_stop=0):
    
    # get the start time of the iteration
    start_time = time.time()
    
    arguments = [x_init, epsilon, bracket_high]

    # call the iterator
    results, iters, total_runtime = iterator(conj_grad_iter, arguments, iterations, early_stop)
    return results, iters, total_runtime 

def conj_grad_iter(x, epsilon, bracket_high):

    # get the start time of the iteration
    start_time = time.time()

    d = -1 * gradient(x)
    y = x
    j = 1
    # perform the inner loop of the method but one time since scalar output
    while np.linalg.norm(gradient(y), ord=2) > epsilon:
        # line 2
        step_size = exact_line_search(y, -1 * d, 
                bracket_high = bracket_high)
        y_next = y + (step_size * d)

        if j == x.shape[0]:
            break

        # line 3
        d = (-1 * gradient(y_next)) + (polak_rebiere(gradient(y_next), gradient(y)) * d)

        y = y_next
        j += 1

    # update x_next
    x_next = y

    return [x_next, epsilon, bracket_high], (time.time() - start_time)




def accelerated_gradient_descent(x_init, iterations, alpha, beta, a, early_stop=0):

    # packed arguments for the iterator
    arguments = [x_init, x_init, alpha, beta, a]

    # call the iterator
    results, iters, total_runtime = iterator(agd_iter, arguments, iterations, early_stop)
    return results, iters, total_runtime 

def agd_iter(x, y, alpha, beta, a):
    
    # get the start time of the iteration
    start_time = time.time()

    grad = gradient(x)
    step_size = inexact_line_search(grad, eval_objective(x), x, alpha, beta)

    # find x_{k+1} (regular gradient step)
    y_next = x - (step_size * gradient(x))
    
    # find a_{k+1} from a_k
    a_next = (1 + math.sqrt(4 * (a**2))) / 2
    #a_next = find_a_next(a, recip_kappa)

    dynamic_momentum = (1 - a) / a_next
    #dynamic_momentum = find_dynamic_momentum(a, a_next)

    # find y_{k_1} (sliding step)
    x_next = y_next + (dynamic_momentum * (y_next - y))

    return [x_next, y_next, alpha, beta, a_next], (time.time() - start_time)



def fista(x_init, iterations, alpha, beta, early_stop=0):

    # packed arguments for the iterator
    arguments = [x_init, x_init, 1, alpha, beta]

    # call the iterator
    results, iters, total_runtime = iterator(fista_iter, arguments, iterations, early_stop)
    return results, iters, total_runtime 

def fista_iter(x, y, t, alpha, beta):
    
    # get the start time of the iteration
    start_time = time.time()
    
    t_next = 0.5 * (1 + math.sqrt(1 + (4 * (t**2))))

    step_size = inexact_line_search(gradient(x), eval_objective(x), x, alpha, beta)

    x_next = y - (step_size * gradient(y))

    y_next = x_next + ( ((t - 1) / t_next) * (x_next - x) )

    return [x_next, y_next, t_next, alpha, beta], (time.time() - start_time)



def barzilai_borwein(x_init, iterations, early_stop=0):

    # packed arguments for the iterator
    arguments = [x_init, None, None]

    # call the iterator
    results, iters, total_runtime = iterator(bb_iter, arguments, iterations, early_stop)
    return results, iters, total_runtime 

def bb_iter(x, x_prev, grad_prev):
    
    # check if it x_0 (k=0th) iteration since no x_prev or f'(x_prev)
    # are available, instead imply perform gradient descent
    if (x_prev is None and grad_prev is None):

        # get the start time of the iteration
        start_time = time.time()

        grad = gradient(x)

        # simple line search with alpha = beta = 0.5
        step_size = inexact_line_search(grad, eval_objective(x), x, 0.5, 0.5)

        # simple gradient descent
        x_next = x - np.dot(step_size, grad)

        return [x_next, x, grad], (time.time() - start_time)

    # get the start time of the iteration
    start_time = time.time()

    grad = gradient(x)
    r = x - x_prev
    q = grad - grad_prev

    step_size = np.dot(r, q) / np.dot(q, q)

    x_next = x - (step_size * grad)

    return [x_next, x, grad], (time.time() - start_time)
