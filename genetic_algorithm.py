import numpy as np

#Hyperparameters
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


def crossover(parents):

    assert (len(parents) == 2)

    length = len(parents[0])
    points = [np.random.randint(0, length, dtype=int) for x in range(2)]
    low, high = min(points), max(points)

    children = np.empty((2, length), dtype=int)

    for i, parent in enumerate(parents):

        other_parent = (parents[0] if i == 1 else parents[1])
        middle = parent[low:high]
        remaining = np.setdiff1d(other_parent, middle, assume_unique=True)

        end = remaining[:length - high]
        start = remaining[length - high:]

        children[i] = np.concatenate((start, middle, end))

    return children




def mutate(individual):
    """
    Mutates a single individual using the displacement mutation (DM) operator. DM was used
    because it was found to be the best mutation operator tested by Larranaga and Kuijpers.
    :param individual: a NumPy array
    :return:
    """

    points = [np.random.randint(0, len(individual), dtype=int) for x in range(2)]
    low, high = min(points), max(points)

    mutation = individual[low: high]
    chromosome = np.concatenate((individual[:low], individual[high:]))
    insertion_point = np.random.randint(0, len(chromosome), dtype=int)

    return np.concatenate((chromosome[:insertion_point], mutation, chromosome[insertion_point:]))


population = create_population(8, 2)
children = crossover(population)

print(population)
print(children)





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
