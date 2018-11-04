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

# Later I can OO them so they have a route and a cost.

def crossover(mother, father):

    assert(len(mother) == len(father))
    length = len(mother)

    points = [np.random.randint(0, length, dtype=int) for x in range(2)]
    low, high = min(points), max(points)

    # Let's do one child first, keep things... simple.
    # Need to think really about which section to use...

    print(mother, father, low, high)

    # Maybe I shouldn pass an array slice.

    daughter = mother[low:high]

    remaining = [city for city in father if city not in daughter]

    print(daughter)
    print(remaining)

    end = remaining[:length - high]
    start = remaining[length - high:]

    # And I've done it. Not bad, huh?

    print(start, daughter, end)

    return np.concatenate((start, daughter, end))


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
child = crossover(population[0], population[1])

print(child)





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
