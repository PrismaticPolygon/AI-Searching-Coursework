import numpy as np

population_size = 200
mutation_probability = 0.01
selective_pressure = 1.90





#Read through document; we'll find the best crossover and mutation operators.

# Path representation
# ER crossover.
# DM, IVM, ISM (efficacy). SIM, SM (speed)
# size of population = 200; probability of mutation (0.01), selective pressure (1.90)

# Representation?
# Define an object?
# Making classes! Wrong choice.
# Doesn't really need because the the matrix; we can just query the look-up table.
# Also more extensible for other algos.
# Cities may as well be row vectors.

city = np.random.choice(5, 5, replace=False) + 1

# Parameters required:

# Matrix / vectorised implementation probably a good shout. I'll move to that later.

print(city)