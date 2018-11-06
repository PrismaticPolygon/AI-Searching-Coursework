import numpy as np
from main import load_file

#Hyperparameters
population_size = 300
mutation_probability = 0.05
crossover_probability = 0.7
num_generations = 1000
size, distance_matrix = load_file("NEWAISearchfile017.txt")

class Individual:

    def __init__(self, chromosome=None):

        self.chromosome = chromosome if chromosome is not None else np.random.choice(size, size, replace=False)
        self.cost, self.fitness = self.generate_fitness()

    @classmethod
    def from_chromosome(cls, chromosome):

        return cls(chromosome)

    def mutate(self):
        points = [np.random.randint(0, len(self.chromosome), dtype=int) for x in range(2)]
        low, high = min(points), max(points)

        mutation = self.chromosome[low: high]
        new_chromosome = np.concatenate((self.chromosome[:low], self.chromosome[high:]))
        insertion_point = np.random.randint(0, len(new_chromosome), dtype=int)

        self.chromosome = np.concatenate((new_chromosome[:insertion_point], mutation, new_chromosome[insertion_point:]))
        self.cost, self.fitness = self.generate_fitness()

    def generate_fitness(self):

        fitness = distance_matrix[self.chromosome[0], self.chromosome[-1]]

        for i in range(1, size):

            fitness += distance_matrix[self.chromosome[i - 1], self.chromosome[i]]

        self.cost = fitness

        return fitness, 1 /  fitness

    def __str__(self):

        return str(self.fitness ** -1) + ": " + ", ".join(map(str, self.chromosome))


class Population:

    def __init__(self):

        individuals = []
        # individuals = np.empty(shape=(population_size, size))

        for i in range(population_size):

                individuals.append(Individual())
        #     individuals[i, :] = Individual()

        self.individuals = individuals
        self.current_generation = 0
        self.best_individual = self.worst_individual = individuals[0]

    def crossover(self, parents):

        points = [np.random.randint(0, size, dtype=int) for x in range(2)]
        low, high = min(points), max(points)

        #children = np.empty((2, size), dtype=int)
        children = []

        for i, parent in enumerate(parents):
            other_parent = (parents[0] if i == 1 else parents[1])
            middle = parent.chromosome[low:high]
            remaining = np.setdiff1d(other_parent.chromosome, middle, assume_unique=True)

            end = remaining[:size - high]
            start = remaining[size - high:]

            #children[i] = Individual.from_chromosome(np.concatenate((start, middle, end)))
            children.append(Individual.from_chromosome(np.concatenate((start, middle, end))))

        return children

    def roulette_selection(self):

        fitness_sum = sum([i.fitness for i in self.individuals])
        parents = []

        def spin():

            temp = 0
            rand = np.random.uniform(0, fitness_sum)

            for i in self.individuals:

                temp += i.fitness

                if temp > rand:

                    return i

        for x in range(population_size):

            parents.append(spin())

        return parents

    def rank_selection(self):

        parents = []
        pool = sorted(self.individuals, key=lambda i: i.fitness)
        rank_sum = int(((population_size + 1) * population_size) / 2) - population_size

        def rank():

            temp = 0
            rand = np.random.randint(0, rank_sum)

            for i in range(len(pool) - 1, 0, -1):

                temp += i

                if temp >= rand:

                    return pool[i]

        for x in range(population_size):

            parents.append(rank())

        return parents

    def select(self, selection_type):

        if selection_type is None:

            return self.roulette_selection()

        if selection_type == "roulette":

            return self.roulette_selection()

        elif selection_type == "rank":

            return self.rank_selection()

    def evolve(self):

        # Somehow I have 20 parents and only 6 individuals? What's going on?

        parents = self.select('rank')
        children = []

        for i in range(1, population_size, 2):

            if np.random.rand() < crossover_probability:

                children += self.crossover(parents[i - 1: i + 1])

            else:

                children += parents[i - 1: i + 1]

        for child in children:

            if np.random.rand() < mutation_probability:

                child.mutate()

        self.individuals = children

        potential_best = max(self.individuals, key=lambda i: i.fitness)

        if potential_best.fitness > self.best_individual.fitness:

            self.best_individual = potential_best

        potential_worst = min(self.individuals, key=lambda i: i.fitness)

        if potential_worst.fitness > self.worst_individual.fitness:

            self.worst_individual = potential_worst

    def __str__(self):

        return "\n".join(map(str, self.individuals))


population = Population()

for i in range(num_generations):

    population.evolve()

    print("Best: " + str(population.best_individual))
    print("Worst: " + str(population.worst_individual))

    # It's improved, but not by much.


# Right: fitness does not seem to be improving.



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
