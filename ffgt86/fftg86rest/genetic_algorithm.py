import numpy as np


class GeneticAlgorithm:

    def __init__(self, distance_matrix, length, mutation_probability=0.35,
                 crossover_probability=0.55, tournament_size=6, num_generations=1000, population_size=200):

        self.distance_matrix = distance_matrix
        self.length = length
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.tournament_size = tournament_size
        self.num_generations = num_generations

        self.population = np.zeros((population_size, length), dtype=int)

        for k in range(population_size):

            self.population[k] = np.random.choice(length, length, replace=False)

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

    def select(self):

        competitors = np.random.choice(self.population_size, self.tournament_size, replace=False)

        return self.population[min(competitors, key=lambda i: self.get_cost(i))]

    def pmx(self, parents):

        children = np.empty((2, self.length), dtype=int)
        points = [np.random.randint(0, self.length) for x in range(2)]
        low, high = min(points), max(points)

        for j in range(2):

            mother, father = parents[j % 2], parents[(j + 1) % 2]

            child = np.full(mother.shape, -1, dtype=int)
            child[low:high] = mother[low:high]

            for i in range(low, high):

                val = father[i]

                if val not in child:

                    def internal():

                        k, temp = i, val

                        while True:

                            i_mother = np.where(father == mother[k])[0][0]

                            if i_mother < low or i_mother >= high:

                                return i_mother

                            else:

                                k, temp = i_mother, mother[k]

                    index = internal()
                    child[index] = val

            for (i, val) in enumerate(child):

                if val == -1:

                    child[i] = father[i]

            children[j] = child

        return children

    def ox1(self, parents):

        points = [np.random.randint(0, self.length) for x in range(2)]
        low, high = min(points), max(points)
        children = np.empty((2, self.length), dtype=int)

        for j in range(2):

            mother, father = parents[j % 2], parents[(j + 1) % 2]
            middle = mother[low:high]
            remaining = np.setdiff1d(father, middle, assume_unique=True)

            end = remaining[:self.length - high]
            start = remaining[self.length - high:]

            children[j] = np.concatenate((start, middle, end))

        return children

    def breed(self):

        children = np.empty((self.population_size, self.length), dtype=int)

        for i in range(0, self.population_size, 2):

            parents = np.array([self.select(), self.select()])

            if self.crossover_probability >= np.random.rand():

                children[i:i+2] = self.ox1(parents)

            else:

                children[i:i+2] = parents

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

        if self.best_generation == 0:

            self.best_generation = self.num_generations

        return self.best_route + 1, self.best_cost
