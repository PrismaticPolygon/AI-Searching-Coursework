import numpy as np
from main import load_file, write_file

filename = "NEWAISearchfile042.txt"
length, distance_matrix = load_file(filename)

# Hyperparameters
population_size = 1000
mutation_probability = 0.05
crossover_probability = 0.7
tournament_size = 5
num_generations = 1000

np.random.seed(1)

# Now, how am I going to experiment with my hyperparameters?
# Even worse, the mean cost actually increased. Outrageous.
# Let's modify to store the best solution, shall we?
# Or we could store the max, min, and mean fitnesses.
# I can't believe that it's increasing...

# The greater the tournament size, the greater the selection pressure: weak individuals have a smaller chance to be
# selected, and strong individuals are likely to be selected multiples.

# efficient to code, works on parallel architectures, allows for selection pressure to be easily adjusted,
# independent of the scaling of the GA fitness function

# Best is still only around 1974

# Convergence check.


class GeneticAlgorithm:

    def __init__(self):

        self.population = np.zeros((population_size, length), dtype=int)

        for i in range(population_size):

            self.population[i] = np.random.choice(length, length, replace=False)

    def get_mean_cost(self):

        sum = 0

        for i in range(population_size):

            sum += self.get_cost(i)

        return sum / population_size

    def get_best(self):

        best_index = min([x for x in range(population_size)], key=lambda i: self.get_cost(i))
        best_cost = self.get_cost(best_index)

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

            # Herein lies the problem. But why?

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

            print(str(self.get_mean_cost()), str(cost))

        return self.get_best()


ga = GeneticAlgorithm()

tour, cost = ga.evolve()

write_file(filename, "A", tour + 1, cost)
#
# print(best)
