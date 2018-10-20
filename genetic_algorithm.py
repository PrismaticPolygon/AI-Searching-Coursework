import numpy as np

population_size = 200
mutation_probability = 0.01
selective_pressure = 1.90


def create_population(num_cities, population_size):
    """
        Creates a new population of individuals, a matrix of size population_size x num_cities.
        Each row of the matrix corresponds to an individual.
    :param num_cities:
    :param population_size:
    :return:
    """

    cities = np.empty(shape=(population_size, num_cities), dtype=int)

    for i in range(population_size):

        cities[i, :] = np.random.choice(num_cities, num_cities, replace=False) + 1

    return cities


def mutate(individual):
    """
    Mutates a single individual using the displacement mutation (DM) operator. DM was used
    because it was found to be the best mutation operator tested by Larranaga and Kuijpers.
    :param individual: a NumPy array
    :return:
    """

    length = len(individual)

    a = np.random.randint(0, length, dtype=int)
    b = np.random.randint(0, length, dtype=int)
    low = min(a, b)
    high = max(a, b)

    subtour = individual[low: high]
    insertion_point = np.random.randint(0, length - len(subtour), dtype=int)

    new_individual = np.concatenate((individual[:insertion_point], subtour, individual[insertion_point + len(subtour):]))

    return new_individual


population = create_population(8, 10)
mutate(population[0])





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


# Parameters required:

# Matrix / vectorised implementation probably a good shout. I'll move to that later.
