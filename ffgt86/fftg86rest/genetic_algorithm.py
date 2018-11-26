import numpy as np


class GeneticAlgorithm:

    def __init__(self, distance_matrix, length, population_size=100, mutation_probability=0.05,
                 crossover_probability=0.7, tournament_size=3, num_generations=2000):

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

    def get_average(self):

        cost_sum = 0

        for i in range(self.population_size):

            cost_sum += self.get_cost(i)

        return cost_sum / self.population_size

    def get_selection_pressure(self):

        # About 1.1 is desirable, apparently.

        return self.get_average() / self.get_best()[1]

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

    # Tournament selection currently used in conjunction with 'noisy' fitness functions.
    # The selection pressure is the degree to which the better individuals are favoured.
    # Increase selection pressure by increasing the tournament size: winner will on average have a higher fitness
    # than the winner of a smaller tournament.
    # This could be very useful.
    # Maybe I should return to rank.
    # Ooh that's very cool: given the current population fitness mean and variance, can predict
    # the average population fitness and iteratively to predict the convergence rate of the GA.
    # http://wpmedia.wolfram.com/uploads/sites/13/2018/02/09-3-2.pdf

    def select(self):

        competitors = np.random.choice(self.population_size, self.tournament_size, replace=False)

        return self.population[min(competitors, key=lambda i: self.get_cost(i))]

    def pmx(self, parents):

        points = [np.random.randint(0, self.length) for x in range(2)]
        low, high = min(points), max(points)

        child = np.full(parents[0].shape, -1, dtype=int)


    def ox1(self, parents):

        points = [np.random.randint(0, self.length) for x in range(2)]
        low, high = min(points), max(points)
        children = np.empty((2, self.length), dtype=int)

        for j, parent in enumerate(parents):

            other_parent = (parents[0] if j == 1 else parents[1])
            middle = parent[low:high]
            remaining = np.setdiff1d(other_parent, middle, assume_unique=True)

            end = remaining[:self.length - high]
            start = remaining[self.length - high:]

            children[j] = np.concatenate((start, middle, end))

        return children

    def breed(self):

        children = np.empty((self.population_size, self.length), dtype=int)

        for i in range(0, self.population_size, 2):

            parents = [self.select(), self.select()]

            if self.crossover_probability >= np.random.rand():

                children[i:i+2] = self.ox1(parents)

            else:

                children[i], children[i + 1] = parents[0], parents[1]

        self.population = children

    def evolve(self):

        for i in range(self.num_generations):

            self.breed()
            self.mutate()

            best, cost = self.get_best()

            # But that's not what we're doing, is it?
            # That's per-locus: every bit has that probability. Cumulatively, does that guarantee mutation?
            # Selection pressure has definitely changed... weird.

            if cost < self.best_cost:

                self.best_route = best
                self.best_cost = cost
                self.best_generation = i

                print("New best (gen = {}, pressure = {:.4f}):".format(i, self.get_selection_pressure()), self.best_cost)

        return self.best_route + 1, self.best_cost


# https://pdfs.semanticscholar.org/39f0/b09c38f60537ee28eb836a51466d0cd1a787.pdf
# https://en.wikipedia.org/wiki/Genetic_algorithm
