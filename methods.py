import numpy as np

# create an x of length n
def init_x(n):
    x = np.ones((n))
    x[::2] -= 2.2
    return x

# evaluate the function at x
def eval_obj(x):
    val = 0
    for i in range(1, x.shape[0]):
        val += 100 * ((x[i] - ((x[i-1])**2))**2) + ((1 - x[i-1])**2)
    return val

# the gradient at some particular x_i and x_{i-1}
def gradient(x):
    grad = np.zeros_like(x)

    for i in range(1, x.shape[0]):
        grad[i] += d_x(x[i], x[i-1])
        grad[i-1] += d_ximo(x[i], x[i-1])

    return grad

def line_search():
    #TODO

# derivative with respect to x_i
def d_x(x_i, x_imo):
    #TODO

# derivative with respect to x_i-1
def d_ximo(x_i, x_imo):
    #TODO

def gradient_descent(x, a):
    x -= np.dot(a, gradient(x))
    return x

def heavy_ball(x_k, x_kmo, a, b):
    # gradient
    x -= np.dot(a, gradient(x)) 
    # momentum
    x += np.dot(b, (x_k - x_kmo))
    return x

def p_r_conjugate_gradient():
    #TODO

def n_acc_gradient_descent():
    #TODO

def fista():
    #TODO

def barzilai_borwein():
    #TODO
