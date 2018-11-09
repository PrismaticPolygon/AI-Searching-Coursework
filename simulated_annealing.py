from main import write_file, get_files
import numpy as np
import math

# Hyperparameters
min_temp = 0.0001
temp = 1
alpha = 0.999995


class SimulatedAnnealing:

    def __init__(self):

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

        self.best_route = self.route = route

    def get_cost(self, route):

        cost = distance_matrix[route[-1], route[0]]

        for i in range(1, length):

            cost += distance_matrix[route[i - 1], route[i]]

        return cost

    def generate_neighbour(self):

        neighbour_difference = math.ceil((length - 2) * temp) + 1

        index = np.random.randint(1, length - neighbour_difference + 1)

        neighbour = np.copy(self.route)

        np.random.shuffle(neighbour[index: neighbour_difference])

        return neighbour

    def temperature_schedule(self):

        global temp, min_temp, alpha

        while temp > min_temp:
            temp *= alpha

            yield temp

    def accept(self, route):

        current_cost, new_cost = self.get_cost(self.route), self.get_cost(route)

        if new_cost < current_cost:

            return 1

        return math.exp(-abs(current_cost - new_cost) / temp)

    def anneal(self):

        print("Initial route: ", self.get_cost(self.route))

        for temp in self.temperature_schedule():

            neighbour = self.generate_neighbour()

            if self.accept(neighbour) > np.random.rand():

                self.route = neighbour

                if self.get_cost(neighbour) < self.get_cost(self.best_route):

                    self.best_route = neighbour

                    print("New best: ", self.get_cost(self.best_route))

        return self.best_route, self.get_cost(self.best_route)


for filename, (length, distance_matrix) in get_files():

    sa = SimulatedAnnealing()
    tour, cost = sa.anneal()

    write_file(filename, "B", tour + 1, cost)

    print("\n")


# http://katrinaeg.com/simulated-annealing.html
# https://github.com/chncyhn/simulated-annealing-tsp/blob/master/anneal.py
# http://www.psychicorigami.com/2007/06/28/tackling-the-travelling-salesman-problem-simmulated-annealing/
# https://arxiv.org/pdf/cs/0001018.pdf (Adaptive Simulated Annealing)
# https://www.ingber.com/ASA-README.html
# https://www.sciencedirect.com/science/article/pii/S0304414901000825
