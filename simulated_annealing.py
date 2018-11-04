from main import load_file
import numpy as np
import math


size, distance_matrix = load_file("NEWAISearchfile048.txt")

# Hyperparameters
min_temp = 0.001
temp = 1
alpha = 0.995
comparison_size = 450

# This feels so much like it should be a GA. We're wasting so much information with each iteration.
# That'd be cool: rank solutions by how far away from each other they are, then their fitness.
# It'd be a huge dictionary of data points (distance from centre, cost), where the centre would be 1, 2, 3, 4, 5 ...


class Solution:

    def generate_route(self):

        return np.random.choice(size, size, replace=False)

    def generate_cost(self):

        cost = distance_matrix[self.route[0], self.route[-1]]

        for i in range(1, size):

            cost += distance_matrix[self.route[i - 1], self.route[i]]

        return cost

    def __init__(self):

        self.route = self.generate_route()
        self.cost = self.generate_cost()

    def __str__(self):

        return str(self.cost) + ": " + " -> ".join(map(str, self.route))


class GreedySolution(Solution):

    def generate_route(self):

        route = [np.random.choice(size)]

        while len(route) < size:

            min, min_index = math.inf, 0

            for i, next in enumerate(distance_matrix[route[-1]]):

                if i not in route and next < min:

                    min, min_index = next, i

            route.append(min_index)

        return np.array(route)


def acceptance_probability(old, new):

    if new.cost < old.cost:

        return 1

    return math.exp(-abs(old.cost - new.cost) / temp)

# Greedy initial solution. Good thinking.
# Does it provide that much of a benefit, though?

# Maybe I should be generating successors differently? I'm just doing it randomly at the moment, in truth.
# Dynamically change the number of iterations as the algorithm progresses: at higher temperatures, fewer iterations.

# Only reduce temperature when a better solution is found...
# And what does he mean, 'neighbours'? In what sense are they neighbouring? Am I only changing one aspect?


def anneal():

    global min_temp, temp, alpha, comparison_size

    best_solution = solution = GreedySolution()

    print("Initial solution: ", solution)

    print("\nCommencing annealing...")

    while temp > min_temp:

        i = 1

        while i < comparison_size:

            new_solution = Solution()

            if acceptance_probability(solution, new_solution) > np.random.rand():

                solution = new_solution

                if new_solution.cost < best_solution.cost:

                    best_solution = new_solution

                    print("New best: " + str(best_solution))

            i += 1

        temp *= alpha

        print(temp)

    return best_solution


print(anneal())


# http://katrinaeg.com/simulated-annealing.html
# https://github.com/chncyhn/simulated-annealing-tsp/blob/master/anneal.py
# http://www.psychicorigami.com/2007/06/28/tackling-the-travelling-salesman-problem-simmulated-annealing/
# https://arxiv.org/pdf/cs/0001018.pdf (Adaptive Simulated Annealing)
