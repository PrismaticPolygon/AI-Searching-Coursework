import numpy as np
from main import load_file, get_files

class Individual:

    def __init__(self, chromosome=None):

        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = np.random.choice(chromosome_length, chromosome_length, replace=False) + 1

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

        cost = distance_matrix[self.chromosome[0] - 1, self.chromosome[-1] - 1]

        for i in range(1, chromosome_length):

            cost += distance_matrix[self.chromosome[i - 1] - 1, self.chromosome[i] - 1]

        return cost, 1 / cost

    def __str__(self):

        return str(self.fitness ** -1) + ": " + ", ".join(map(str, self.chromosome))


class Population:

    def __init__(self, population_size, mutation_probability, crossover_probability):

        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability

        individuals = []
        # individuals = np.empty(shape=(population_size, size))

        for i in range(population_size):
            individuals.append(Individual())
        #       individuals[i, :] = Individual()

        self.individuals = individuals
        self.current_generation = 0
        self.best_individual = self.worst_individual = individuals[0]

    def crossover(self, parents):

        points = [np.random.randint(0, chromosome_length, dtype=int) for x in range(2)]
        low, high = min(points), max(points)

        # children = np.empty((2, size), dtype=int)
        children = []

        for i, parent in enumerate(parents):
            other_parent = (parents[0] if i == 1 else parents[1])
            middle = parent.chromosome[low:high]
            remaining = np.setdiff1d(other_parent.chromosome, middle, assume_unique=True)

            end = remaining[:chromosome_length - high]
            start = remaining[chromosome_length - high:]

            # children[i] = Individual.from_chromosome(np.concatenate((start, middle, end)))
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

        for x in range(self.population_size):
            parents.append(spin())

        return parents

    def rank_selection(self):

        parents = []
        pool = sorted(self.individuals, key=lambda i: i.cost, reverse=True)
        rank_sum = int(((self.population_size + 1) * self.population_size) / 2) - self.population_size

        def rank():

            temp = 0
            rand = np.random.randint(0, rank_sum)

            for i in range(len(pool) - 1, 0, -1):

                temp += i

                if temp >= rand:
                    return pool[i]

        for x in range(self.population_size):
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

        for i in range(1, self.population_size, 2):

            if np.random.rand() < self.crossover_probability:

                children += self.crossover(parents[i - 1: i + 1])

            else:

                children += parents[i - 1: i + 1]

        for child in children:

            if np.random.rand() < self.mutation_probability:

                child.mutate()

        self.individuals = children

        potential_best = max(self.individuals, key=lambda i: i.fitness)

        if potential_best.fitness > self.best_individual.fitness:
            self.best_individual = potential_best

            #print("New best: " + str(self.best_individual))

        potential_worst = min(self.individuals, key=lambda i: i.fitness)

        if potential_worst.fitness > self.worst_individual.fitness:
            self.worst_individual = potential_worst

    def __str__(self):

        return "\n".join(map(str, self.individuals))


class GeneticAlgorithm:

    def __init__(self, population_size=-1, mutation_probability=-1.0, crossover_probability=-1.0, num_generations=-1):

        self.population_size = 100 if population_size == -1 else population_size
        self.mutation_probability = 0.05 if mutation_probability == -1 else mutation_probability
        self.crossover_probability = 0.7 if crossover_probability == -1 else crossover_probability
        self.num_generations = 1000 if num_generations == -1 else num_generations

        self.population = Population(self.population_size, self.mutation_probability, self.crossover_probability)

    def run(self):

        for i in range(self.num_generations):

            self.population.evolve()

        return self.population.best_individual


for chromosome_length, distance_matrix in get_files():

    data = []

    for population_size in range(10, 1000, 10):

        print((population_size / 1000) * 100)

        for i in range(0, 1):

            mutation_probability = i * 0.01

            for j in range(0, 1):

                crossover_probability = j * 0.01

                for num_generations in range(100, 10000, 100):
                    ga = GeneticAlgorithm(population_size, mutation_probability, crossover_probability, num_generations)

                    best = ga.run()

                    data.append(
                        (population_size, mutation_probability, crossover_probability, num_generations, best.cost))

    with open('genetic_algorithm_data' + str(chromosome_length) + '.txt', 'w') as f:

        for item in data:

            f.write("%s\n" % ", ".join(map(str, item)))



# write_file(filename, population.best_individual.chromosome, population.best_individual.cost)
