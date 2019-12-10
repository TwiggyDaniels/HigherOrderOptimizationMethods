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
SIZE=10

# create x_0 per the assignment instructions
x = init_x(SIZE)


results = {}
# perform all the methods and store them in a dictionary

# Parameters: alpha = 0.5, beta = 0.5
results['Gradient Descent'] = gradient_descent(x, ITERATIONS, 0.270848749, 0.874143749, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5, momentum = 0.5
results['Heavy Ball'] = heavy_ball(x, ITERATIONS, 0.020843749, 0.291387916, 0.291679583, early_stop=EARLY_STOP)

# Parameters: bracket_high = 0.5, epsilon=1E-5
results['Conjugate Gradient'] = conjugate_gradient(x, ITERATIONS, 0.458347916, epsilon=0.1, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5, a_0 = 0.5
results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
        ITERATIONS, 0.0833449, 0.2497625, 0.0416, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5
results['FISTA'] = fista(x, ITERATIONS, 0.250015, 0.45788958333333335, early_stop=EARLY_STOP)

# Parameters: NONE
results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

# plot all of the results
display_results(results, str(SIZE))

