import numpy as np


class GeneticAlgorithm:

    def __init__(self, distance_matrix, length, population_size=150, mutation_probability=0.1,
                 crossover_probability=0.8, tournament_size=8, num_generations=500):

        self.distance_matrix = distance_matrix
        self.length = length
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.tournament_size = tournament_size
        self.num_generations = num_generations

        self.population = np.zeros((population_size, length), dtype=int)

        for k in range(population_size):

            route = np.full(length, -1, dtype=int)
            route[0] = np.random.choice(length)

            for i in range(1, length):

                start, next = route[i - 1], None

                for j in range(length):

                    if j not in route:

                        if next is None:

                            next = j

                        elif distance_matrix[start, j] < distance_matrix[start, next]:

                            next = j

                route[i] = next

            self.population[k] = route
            # self.population[k] = np.random.choice(length, length, replace=False)

        self.best_route, self.best_cost = self.get_best()
        self.best_generation = 0

    def get_best(self):

        best_index = min([x for x in range(self.population_size)], key=lambda i: self.get_cost(i))

        return self.population[best_index], self.get_cost(best_index)

    def get_cost(self, i):

        individual = self.population[i]

        cost = self.distance_matrix[individual[-1], individual[0]]

        for i in range(1, len(individual)):

            cost += self.distance_matrix[individual[i - 1], individual[i]]

        return cost

    def mutate(self):

        for i in range(self.population_size):

            if self.mutation_probability >= np.random.rand():

                points = [np.random.randint(0, self.length) for x in range(2)]
                low, high = min(points), max(points)
                insert = np.random.randint(0, self.length - high + low)

                subtour = self.population[i][low: high]
                new = np.concatenate((self.population[i][:low], self.population[i][high:]))

                self.population[i] = np.concatenate((new[:insert], subtour, new[insert:]))

    def breed(self):

        children = np.empty((self.population_size, self.length), dtype=int)

        def select():

            competitors = np.random.choice(self.population_size, self.tournament_size, replace=False)

            return self.population[min(competitors, key=lambda i: self.get_cost(i))]

        def crossover(parents):

            points = [np.random.randint(0, self.length) for x in range(2)]
            low, high = min(points), max(points)

            for j, parent in enumerate(parents):

                other_parent = (parents[0] if j == 1 else parents[1])
                middle = parent[low:high]
                remaining = np.setdiff1d(other_parent, middle, assume_unique=True)

                end = remaining[:self.length - high]
                start = remaining[self.length - high:]

                children[i + j] = np.concatenate((start, middle, end))

        for i in range(0, self.population_size, 2):

            parents = [select(), select()]

            if self.crossover_probability >= np.random.rand():

                crossover([select(), select()])

            else:

                children[i], children[i + 1] = parents[0], parents[1]

        self.population = children

    def evolve(self):

        for i in range(self.num_generations):

            self.breed()
            self.mutate()

            best, cost = self.get_best()

            if cost < self.best_cost:

                self.best_route = best
                self.best_cost = cost
                self.best_generation = i

                print("New best (gen = {}):".format(i), self.best_cost)

        return self.best_route + 1, self.best_cost


# https://pdfs.semanticscholar.org/39f0/b09c38f60537ee28eb836a51466d0cd1a787.pdf
# https://en.wikipedia.org/wiki/Genetic_algorithm
