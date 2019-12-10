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
SIZE=10

SEARCHES=24

# create x_0 per the assignment instructions
x = init_x(SIZE)

def build_grid(mins_maxes, searches):
    ranges = np.zeros((mins_maxes.shape[0], searches))

    for i, _ in enumerate(mins_maxes):
        # get the step size
        inc = (mins_maxes[i][0] + mins_maxes[i][1]) / searches
        # set the mins
        ranges[i][0] = mins_maxes[i][0]
        ranges[i][searches-1] = mins_maxes[i][1]
        for v in range(1, searches - 1):
            ranges[i][v] = ranges[i][0] + (v * inc)
    return ranges

print("Grid Search for n=" + str(SIZE))

results = {}

# find the parameters for the grid search
# Parameters: alpha = (0, 0.5], beta = (0, 1)
gd_grid = build_grid(np.array([[1E-5, 0.5], [1E-5, 0.999]]), SEARCHES)
counter = 0
best = None
best_idx = None
for i, a in enumerate(gd_grid[0]):
    for j, b in enumerate(gd_grid[1]):
        tmp = gradient_descent(x, ITERATIONS, a, b, early_stop=EARLY_STOP)
        if best is None:
            best = tmp
        else:
            if tmp[0][tmp[1]] <= best[0][best[1]]:
                # take the one with a lower number of convergence iterations
                best = tmp
                best_idx = [a, b]
        counter += 1

print("GD:", best_idx)
results['Gradient Descent'] = best

# Parameters: alpha = (0, 0.5], beta = (0, 1), a_0 = [0.0, 1.0]
agd_grid = build_grid(np.array([[1E-5, 0.5], [1E-5, 0.999], [0.0, 1.0]]), SEARCHES)
counter = 0
best = None
best_idx = None
for i, a in enumerate(agd_grid[0]):
    for j, b in enumerate(agd_grid[1]):
        for k, c in enumerate(agd_grid[2]):
            tmp = accelerated_gradient_descent(x, ITERATIONS, a, b, c, early_stop=EARLY_STOP)
            if best is None:
                best = tmp
            else:
                if tmp[0][tmp[1]] <= best[0][best[1]]:
                    # take the one with a lower number of convergence iterations
                    best = tmp
                    best_idx = [a, b, c]
            counter += 1

print("ACC:", best_idx)
results['Accelerated Gradient Descent'] = best

# Parameters: alpha = (0, 0.5], beta = (0, 1), momentum = (0, 1]
hb_grid = build_grid(np.array([[1E-5, 0.5], [1E-5, 0.999], [1E-5, 1.0]]), SEARCHES)
counter = 0
best = None
best_idx = None
for i, a in enumerate(hb_grid[0]):
    for j, b in enumerate(hb_grid[1]):
        for k, c in enumerate(hb_grid[2]):
            tmp = heavy_ball(x, ITERATIONS, a, b, c, early_stop=EARLY_STOP)
            if best is None:
                best = tmp
            else:
                if tmp[0][tmp[1]] <= best[0][best[1]]:
                    # take the one with a lower number of convergence iterations
                    best = tmp
                    best_idx = [a, b, c]
            counter += 1

print("HB:", best_idx)
results['Heavy Ball'] = best

# Parameters: bracket_high = (0, 1], epsilon = (0, 1E-1]
cg_grid = build_grid(np.array([[1E-5, 1.0], [1E-5, 1E-1]]), SEARCHES)
counter = 0
best = None
best_idx = None
for i, a in enumerate(cg_grid[0]):
    for j, b in enumerate(cg_grid[1]):
        tmp = conjugate_gradient(x, ITERATIONS, a, b, early_stop=EARLY_STOP)
        if best is None:
            best = tmp
        else:
            if tmp[0][tmp[1]] <= best[0][best[1]]:
                # take the one with a lower number of convergence iterations
                best = tmp
                best_idx = [a, b]
        counter += 1

print("CG:", best_idx)
results['Conjugate Gradient'] = best

# Parameters: alpha = (0, 0.5], beta = (0, 1)
f_grid = build_grid(np.array([[1E-5, 0.5], [1E-5, 0.999]]), SEARCHES)
counter = 0
best = None
best_idx = None
for i, a in enumerate(f_grid[0]):
    for j, b in enumerate(f_grid[1]):
        tmp = fista(x, ITERATIONS, a, b, early_stop=EARLY_STOP)
        if best is None:
            best = tmp
        else:
            if tmp[0][tmp[1]] <= best[0][best[1]]:
                # take the one with a lower number of convergence iterations
                best = tmp
                best_idx = [a, b]
            counter += 1

print("FISTA:", best_idx)
results['FISTA'] = best

# Parameters: NONE
results['Barzilai Borwein'] = barzilai_borwein(x, ITERATIONS, early_stop=EARLY_STOP)

# plot all of the results
display_results(results, str(SIZE))
