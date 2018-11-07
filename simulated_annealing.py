from main import load_file, write_file
import numpy as np
import math

# Hyperparameters
min_temp = 0.001
temp = 1
alpha = 0.995
neighbourhood_size = 300
filename = "NEWAISearchfile017.txt"
size, distance_matrix = load_file(filename)

# This feels so much like it should be a GA. We're wasting so much information with each iteration.
# That'd be cool: rank solutions by how far away from each other they are, then their fitness.
# It'd be a huge dictionary of data points (distance from centre, cost), where the centre would be 1, 2, 3, 4, 5 ...
# This can't be how it is implemented, just can't.

# Aha! We're not CONSERVATIVELY generating neighbours: this truly is purely random search.
# Neighbours are the set of permutations produced by reversing the order of any two successive cities!
# That makes a ton more sense.

# Return temperature as a generator?


class Solution:

    def generate_route(self):

        return np.random.choice(size, size, replace=False) + 1

    def generate_cost(self):

        cost = distance_matrix[self.route[0] - 1, self.route[-1] - 1]

        for i in range(1, size):

            cost += distance_matrix[self.route[i - 1] - 1, self.route[i] - 1]

        return cost

    def __init__(self, route=None):

        self.route = route if route is not None else self.generate_route()
        self.cost = self.generate_cost()

    @classmethod
    def from_route(cls, route):

        return cls(route)

    def generate_neighbours(self):

        i = 0

        while i < neighbourhood_size:

            index = np.random.randint(1, size)

            neighbour_route = np.copy(self.route)
            neighbour_route[index], neighbour_route[index - 1] = neighbour_route[index - 1], neighbour_route[index]

            yield Solution.from_route(neighbour_route)

            i += 1

    def __str__(self):

        return str(self.cost) + ": " + " -> ".join(map(str, self.route))


class GreedySolution(Solution):

    def generate_route(self):

        route = np.zeros(size, dtype=int)
        route[0] = np.random.choice(size) + 1

        for i in range(1, size):

            min, min_index, cur_index = math.inf, 0, route[i - 1] - 1

            for j in range(size):

                cost = distance_matrix[cur_index, j]

                if j == cur_index:  # The case where the index points to itself.

                    continue

                if (j + 1) not in route and cost < min:

                    min, min_index = cost, (j + 1)

            route[i] = min_index

        return route


def acceptance_probability(old, new):

    if new.cost < old.cost:

        return 1

    return math.exp(-abs(old.cost - new.cost) / temp)


def temperature_schedule():

    global temp, min_temp, alpha

    while temp > min_temp:

        temp *= alpha

        yield temp

# Greedy initial solution. Good thinking.
# Does it provide that much of a benefit, though?

# Maybe I should be generating successors differently? I'm just doing it randomly at the moment, in truth.
# Dynamically change the number of iterations as the algorithm progresses: at higher temperatures, fewer iterations.

# Only reduce temperature when a better solution is found...
# And what does he mean, 'neighbours'? In what sense are they neighbouring? Am I only changing one aspect?


def anneal():

    best_solution = solution = GreedySolution()

    print(solution)

    print("\nCommencing annealing...\n")

    for temp in temperature_schedule():

        for neighbour in solution.generate_neighbours():

            if acceptance_probability(solution, neighbour) > np.random.rand():

                solution = neighbour

                if neighbour.cost < best_solution.cost:

                    best_solution = neighbour

                    print("New best: " + str(best_solution))

    return best_solution


result = anneal()

print(result)

# write_file(filename, result.route, result.cost)


# http://katrinaeg.com/simulated-annealing.html
# https://github.com/chncyhn/simulated-annealing-tsp/blob/master/anneal.py
# http://www.psychicorigami.com/2007/06/28/tackling-the-travelling-salesman-problem-simmulated-annealing/
# https://arxiv.org/pdf/cs/0001018.pdf (Adaptive Simulated Annealing)
