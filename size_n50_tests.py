import numpy as np
import matplotlib as plt

from helpers import init_x, display_results
from methods import gradient_descent, heavy_ball, barzilai_borwein, \
    accelerated_gradient_descent, conjugate_gradient, fista

# number of iterations to perform
ITERATIONS = 16
# early stopping error (usually doesn't work)
EARLY_STOP=1E-4
# SIZE of the input vectors
SIZE=50

# create x_0 per the assignment instructions
x = init_x(SIZE)


results = {}
# perform all the methods and store them in a dictionary

# Parameters: alpha = 0.5, beta = 0.5
results['Gradient Descent'] = gradient_descent(x, ITERATIONS, 0.3500169, 0.999, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5, momentum 
results['Heavy Ball'] = heavy_ball(x, ITERATIONS, 0.300016, 0.999, 0.2200012, early_stop=EARLY_STOP)

# Parameters: bracket_high = 0.5, epsilon=1E-5
results['Conjugate Gradient'] = conjugate_gradient(x, ITERATIONS, 1.0, epsilon=1E-5, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5, a_0 = 0.5
results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
        ITERATIONS, 0.200014, 0.594159, 0.0, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5
results['FISTA'] = fista(x, ITERATIONS, 0.5, 0.999, early_stop=EARLY_STOP)

# Parameters: NONE
results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

# plot all of the results
display_results(results, str(SIZE))
