import numpy as np
import matplotlib as plt

from helpers import init_x, display_results
from methods import gradient_descent, heavy_ball, barzilai_borwein, \
    accelerated_gradient_descent, conjugate_gradient, fista

# number of iterations to perform
ITERATIONS = 10
# early stopping error (usually doesn't work)
EARLY_STOP=1E-4
# SIZE of the input vectors
SIZE=5

# create x_0 per the assignment instructions
x = init_x(SIZE)

#
#
#
print("First Experiment for n=5")

results = {}
# perform all the methods and store them in a dictionary

# Parameters: alpha = 0.5, beta = 0.5
results['Gradient Descent'] = gradient_descent(x, ITERATIONS, 0.5, 0.5, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5, momentum = 0.5
results['Heavy Ball'] = heavy_ball(x, ITERATIONS, 0.5, 0.5, 0.5, early_stop=EARLY_STOP)

# Parameters: bracket_high = 0.5, epsilon=1E-5
results['Conjugate Gradient'] = conjugate_gradient(x, ITERATIONS, 1.0, epsilon=1E-5, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5, a_0 = 0.5
results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
        ITERATIONS, 0.5, 0.5, 0.5, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.5
results['FISTA'] = fista(x, ITERATIONS, 0.5, 0.5, early_stop=EARLY_STOP)

# Parameters: NONE
results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

# plot all of the results
display_results(results, str(SIZE))

#
#
#
print("Second Experiment for n=5")

x = init_x(5)
results = {}
# perform all the methods and store them in a dictionary

# Parameters: alpha = 0.5, beta = 0.75
results['Gradient Descent'] = gradient_descent(x, ITERATIONS, 0.5, 0.75, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.75, momentum = 0.4
results['Heavy Ball'] = heavy_ball(x, ITERATIONS, 0.5, 0.75, 0.4, early_stop=EARLY_STOP)

# Parameters: bracket_high = 0.5, epsilon=1E-4
results['Conjugate Gradient'] = conjugate_gradient(x, ITERATIONS, 0.5, epsilon=1E-4, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.75, a_0 = 0.75
results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
        ITERATIONS, 0.5, 0.75, 0.75, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.75
results['FISTA'] = fista(x, ITERATIONS, 0.5, 0.75, early_stop=EARLY_STOP)

# Parameters: NONE
results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

# plot all of the results
display_results(results, str(SIZE))

#
#
#
print("Third Experiment for n=5")

x = init_x(5)
results = {}
# perform all the methods and store them in a dictionary

# Parameters: alpha = 0.5, beta = 0.8
results['Gradient Descent'] = gradient_descent(x, ITERATIONS, 0.5, 0.8, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.6, momentum = 0.4
results['Heavy Ball'] = heavy_ball(x, ITERATIONS, 0.5, 0.6, 0.4, early_stop=EARLY_STOP)

# Parameters: bracket_high = 0.5, epsilon=1E-3
results['Conjugate Gradient'] = conjugate_gradient(x, ITERATIONS, 0.5, epsilon=1E-3, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.75, a_0 = 0.75
results['Accelerated Gradient Descent'] = accelerated_gradient_descent(x, \
        ITERATIONS, 0.5, 0.75, 1.0, early_stop=EARLY_STOP)

# Parameters: alpha = 0.5, beta = 0.75
results['FISTA'] = fista(x, ITERATIONS, 0.5, 0.75, early_stop=EARLY_STOP)

# Parameters: NONE
results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

# plot all of the results
display_results(results, str(SIZE))

