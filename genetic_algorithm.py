import numpy as np
from main import load_file

chromosome_length, distance_matrix = load_file("AISearchtestcase.txt")
population_size = 100
mutation_probability = 0.05
crossover_probability = 0.7
tournament_size = 2
num_generations = 1000


class GeneticAlgorithm:

    def __init__(self):

        self.population = np.zeros((population_size, chromosome_length), dtype=int)

        for i in range(population_size):

            self.population[i] = np.random.choice(chromosome_length, chromosome_length, replace=False)

    def get_cost(self, i):

        individual = self.population[i]

        cost = distance_matrix[individual[0], individual[-1]]

        for i in range(1, len(individual)):

            cost += distance_matrix[individual[i - 1], individual[i]]

        return cost

    def mutate(self):

        for i in range(population_size):

            if mutation_probability <= np.random.rand():

                points = [np.random.randint(0, chromosome_length) for x in range(2)]
                low, high = min(points), max(points)
                insert = np.random.randint(0, chromosome_length - high + low)

                subtour = population[i][low: high]
                new = np.concatenate((population[i][:low], population[i][high:]))

                population[i] = np.concatenate((new[:insert], subtour, new[insert:]))

    def breed(self):

        global population

        children = np.empty((population_size, chromosome_length), dtype=int)

        def select():

            competitors = np.random.choice(population_size, tournament_size, replace=False)

            return self.population[max(competitors, key=lambda i: self.get_cost(i))]

        def crossover(parents):

            points = [np.random.randint(0, chromosome_length) for x in range(2)]
            low, high = min(points), max(points)

            for j, parent in enumerate(parents):

                other_parent = (parents[0] if j == 1 else parents[1])
                middle = parent[low:high]
                remaining = np.setdiff1d(other_parent, middle, assume_unique=True)

                end = remaining[:chromosome_length - high]
                start = remaining[chromosome_length - high:]

                children[i + j] = np.concatenate((start, middle, end))

        for i in range(0, population_size, 2):

            crossover([select(), select()])

        population = children

    def evolve(self):

        for i in range(num_generations):

            self.breed()
            self.mutate()

        best_index = max([x for x in range(population_size)], key=lambda i: self.get_cost(i))

        return self.population[best_index], self.get_cost(best_index)


ga = GeneticAlgorithm()

best = ga.evolve()
print(best)
