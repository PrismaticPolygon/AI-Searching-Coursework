import numpy as np
from main import write_file, get_files

# Hyperparameters
# population_size = 1000
mutation_probability = 0.05
crossover_probability = 0.7
tournament_size = 4
num_generations = 250

# Dynamically modify selection pressure and operators used
# Re-initialise population after convergence
# To regain genetic variation: incest prevention, uniform crossover, favoured replacement of similar individuals,
# segmentation of individuals of similar fitness, increasing population size.


class GeneticAlgorithm:

    def __init__(self):

        self.population = np.zeros((population_size, length), dtype=int)

        for i in range(population_size):

            self.population[i] = np.random.choice(length, length, replace=False)

        self.best_route, self.best_cost = self.get_best()

        print("Initial cost: ", self.best_cost)

    def get_mean_cost(self):

        sum = 0

        for i in range(population_size):

            sum += self.get_cost(i)

        return sum / population_size

    def get_best(self):

        best_index = min([x for x in range(population_size)], key=lambda i: self.get_cost(i))

        return self.population[best_index], self.get_cost(best_index)

    def get_cost(self, i):

        individual = self.population[i]

        cost = distance_matrix[individual[-1], individual[0]]

        for i in range(1, len(individual)):

            cost += distance_matrix[individual[i - 1], individual[i]]

        return cost

    def mutate(self):

        for i in range(population_size):

            if mutation_probability >= np.random.rand():

                points = [np.random.randint(0, length) for x in range(2)]
                low, high = min(points), max(points)
                insert = np.random.randint(0, length - high + low)

                subtour = self.population[i][low: high]
                new = np.concatenate((self.population[i][:low], self.population[i][high:]))

                self.population[i] = np.concatenate((new[:insert], subtour, new[insert:]))

    def breed(self):

        children = np.empty((population_size, length), dtype=int)

        def select():

            competitors = np.random.choice(population_size, tournament_size, replace=False)

            return self.population[min(competitors, key=lambda i: self.get_cost(i))]

        def crossover(parents):

            points = [np.random.randint(0, length) for x in range(2)]
            low, high = min(points), max(points)

            for j, parent in enumerate(parents):

                other_parent = (parents[0] if j == 1 else parents[1])
                middle = parent[low:high]
                remaining = np.setdiff1d(other_parent, middle, assume_unique=True)

                end = remaining[:length - high]
                start = remaining[length - high:]

                children[i + j] = np.concatenate((start, middle, end))

        for i in range(0, population_size, 2):

            parents = [select(), select()]

            if crossover_probability >= np.random.rand():

                crossover([select(), select()])

            else:

                children[i], children[i + 1] = parents[0], parents[1]

        self.population = children

    def evolve(self):

        for i in range(num_generations):

            self.breed()
            self.mutate()

            best, cost = self.get_best()

            if cost < self.best_cost:

                self.best_route = best
                self.best_cost = cost

                print("New best: (gen. {})".format(i), self.best_cost)

        return self.best_route, self.best_cost


for filename, (length, distance_matrix) in get_files():

    population_size = int(0.5 * length) if ((int(0.5 * length) % 2) == 0) else int(0.5 * length) + 1

    print(filename + "\n")

    ga = GeneticAlgorithm()
    tour, cost = ga.evolve()

    write_file(filename, "A", tour + 1, cost)

    print()

