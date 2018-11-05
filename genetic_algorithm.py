import numpy as np
from main import load_file

#Hyperparameters
population_size = 200
mutation_probability = 0.01
num_generations = 100
selective_pressure = 1.90
size, distance_matrix = load_file("AISearchtestcase.txt")

class Individual:

    def mutate(self):

        points = [np.random.randint(0, len(self.chromosome), dtype=int) for x in range(2)]
        low, high = min(points), max(points)

        mutation = self.chromosome[low: high]
        new_chromosome = np.concatenate((self.chromosome[:low], self.chromosome[high:]))
        insertion_point = np.random.randint(0, len(new_chromosome), dtype=int)

        self.chromosome =  np.concatenate((new_chromosome[:insertion_point], mutation, new_chromosome[insertion_point:]))

    def __init__(self):

        self.chromosome = self.generate_chromosome()
        self.cost = self.generate_cost()

    def generate_chromosome(self):

        return np.random.choice(size, size, replace=False) + 1

    def generate_cost(self):

        print(self.chromosome)

        cost = distance_matrix[self.chromosome[0], self.chromosome[-1]]

        for i in range(1, size):

            cost += distance_matrix[self.chromosome[i - 1], self.chromosome[i]]

        return cost


class Population:

    def __init__(self):

        individuals = np.empty(shape=(population_size, size))

        for i in range(population_size):

            individuals[i, :] = Individual()

        self.individuals = individuals

    def crossover(self, parents):

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


population = Population()




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
