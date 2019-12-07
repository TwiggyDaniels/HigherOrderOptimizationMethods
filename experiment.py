import numpy as np
import matplotlib as plt

from helpers import init_x, line_graph#, time_graph
from methods import gradient_descent, heavy_ball, barzilai_borwein, \
    accelerated_gradient_descent, conjugate_gradient, fista

# Inexact Line Search
# Backtracking Constants
ALPHA = 0.5
BETA = 0.5

# initial value a_0 (acc grad desc)
A = 0.9

# estimate of the Lipschitz Constant
LIPSCHITZ = 3

# momentum value
MOMENTUM = 0.2#0.4

# estimate of the Condition Number
KAPPA = 2
RECIPROCAL_KAPPA = 1 / KAPPA

ITERATIONS = 32

EARLY_STOP=1E-2

sizes = [50]


for size in sizes:
    x = init_x(size)
    results = {}
    # perform all the methods and store them in a dictionary
    results['Gradient Descent'] = gradient_descent(x, ITERATIONS, ALPHA, BETA, early_stop=EARLY_STOP)
    results['Heavy Ball'] = heavy_ball(x, ITERATIONS, ALPHA, BETA, MOMENTUM)
    #results['Conjugate Gradient'] = 
    results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
            ITERATIONS, ALPHA, BETA, A, LIPSCHITZ, RECIPROCAL_KAPPA)
    #results['FISTA'] = fista(x, ITERATIONS, LIPSCHITZ)
    results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS)

    line_graph(results, 'Convergences for Size' + str(size))
    #time_graph(results, 'Method Times')
