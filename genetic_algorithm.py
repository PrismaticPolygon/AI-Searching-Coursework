import numpy as np

# Dynamically modify selection pressure and operators used
# Re-initialise population after convergence
# To regain genetic variation: incest prevention, uniform crossover, favoured replacement of similar individuals,
# segmentation of individuals of similar fitness, increasing population size.

# Let's remove the num_generations constraint, and instead calculate convergence.

# FUSS: steadily increase the population size. If memory becomes an issue, delete individuals.


class GeneticAlgorithm:

    def __init__(self, distance_matrix, length, population_size=200, mutation_probability=0.05,
                 crossover_probability=0.7, tournament_size=2, num_generations=1000):

        self.distance_matrix = distance_matrix
        self.length = length
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.tournament_size = tournament_size
        self.num_generations = num_generations

        self.population = np.zeros((population_size, length), dtype=int)

        for i in range(population_size):

            self.population[i] = np.random.choice(length, length, replace=False)

        self.best_route, self.best_cost = self.get_best()
        self.best_generation = 0

    # I am frankly concerned that they haven't converged.
    # How to calculate diversity? Get the mean of the rows, and if it's too close to... something, terminate?
    # Ah, P is pop size. VERY EASY to calculate.

    # Implies that there's a fair bit of variance, none?
    # This is absurd!

    def get_mean_cost(self):

        sum = 0

        for i in range(self.population_size):

            sum += self.get_cost(i)

        return sum / self.population_size

    def mean_individual(self):

        mean = np.sum(self.population, axis=0) / self.population_size

        # Maybe I should sum them up... but why? Get the average per thing.
        # That'll just be half the length, idiota.

        # Then they'll look very different, when in reality they really aren't.

        variance = np.sum(np.sum((self.population - mean) ** 2)) / self.population_size

        # Still an impressive amount of variance... Even after 1000 iterations!
        # It should converge eventually, right? Unless they have the same
        # Oh. Increase tournament size.

        print(variance)

        # How the fuck is there variation with a tournament size of 10? Seriously, what the fuck?
        # Some must have equal fitness, perhaps? Depends on behaviour of max.

        # Sum each column, have summed that row co-ordinate using (x - ci) ^ 2

        # Only works for binary strings, right?

        return mean

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

                print("New best (gen. {}):".format(i), self.best_cost)

        return self.best_route + 1, self.best_cost


    # Optimum results for up to 42 cities! It's got to be done.

    # Store the generation that the best individual was found at.
    # But then I get to tiny mutation rates. OHHHHH, it's per bit! Doesn't make much sense here: cumulative?
    # But then the number of flipped bits will be constant. Solution?
    # And, of course, DM doesn't matter much here: every vertex is connected to every other.
    # As fitness, subtract each tour length from the maximum in the population. Interesting! Shouldn't matter...
    # should it? I doubt it.

    # Fascinating. Try to find k element by element, trying.
    # Only consider sequences of gains whose partial sum is always positive.

# Alternate with hill climbing. Interesting.

# https://pdfs.semanticscholar.org/39f0/b09c38f60537ee28eb836a51466d0cd1a787.pdf
# https://en.wikipedia.org/wiki/Genetic_algorithm

# Should have included references.
