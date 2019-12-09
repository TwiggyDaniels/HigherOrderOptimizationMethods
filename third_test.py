import numpy as np
import matplotlib as plt

from helpers import init_x, plot_results
from methods import gradient_descent, heavy_ball, barzilai_borwein, \
    accelerated_gradient_descent, conjugate_gradient, fista

# Inexact Line Search
# Backtracking Constants
ALPHA = 0.4
BETA = 0.75

# initial value a_0 (acc grad desc)
A = 0.7

# Upper bound for step size in exact line search
BRACKET_HIGH=1.0

# momentum value
MOMENTUM = 0.3

# number of iterations to perform
ITERATIONS = 10

# early stopping error (usually doesn't work)
EARLY_STOP=1E-4

sizes = [5, 10, 50]

for size in sizes:
    x = init_x(size)
    results = {}

    # perform all the methods and store them in a dictionary
    results['Gradient Descent'] = gradient_descent(x, ITERATIONS, ALPHA, BETA, early_stop=EARLY_STOP)
    results['Heavy Ball'] = heavy_ball(x, ITERATIONS, ALPHA, BETA, MOMENTUM, early_stop=EARLY_STOP)
    results['Conjugate Gradient'] = conjugate_gradient(x, ITERATIONS, BRACKET_HIGH, early_stop=EARLY_STOP)
    results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
            ITERATIONS, ALPHA, BETA, A, early_stop=EARLY_STOP)
    results['FISTA'] = fista(x, ITERATIONS, 0.5, 0.9, early_stop=EARLY_STOP)
    results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

    # plot all of the results
    plot_results(results, str(size))
