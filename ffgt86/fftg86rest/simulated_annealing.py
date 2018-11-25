import numpy as np


class SimulatedAnnealing:

    def __init__(self, distance_matrix, length, temp=1, min_temp=0.00001, alpha=0.99995):

        self.distance_matrix = distance_matrix
        self.length = length
        self.temp = temp
        self.min_temp = min_temp
        self.alpha = alpha

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

        cost = self.distance_matrix[route[-1], route[0]]

        for i in range(1, self.length):

            cost += self.distance_matrix[route[i - 1], route[i]]

        return cost

    def generate_neighbour(self):

        points = [np.random.randint(0, self.length) for x in range(2)]
        low, high = min(points), max(points)

        neighbour = np.copy(self.route)

        neighbour[low: high] = neighbour[low: high][::-1]

        return neighbour

    def accept(self, route):

        current_cost, new_cost = self.get_cost(self.route), self.get_cost(route)

        if new_cost < current_cost:

            return 1

        return np.exp((current_cost - new_cost) / self.temp)

    def temperature_schedule(self):

        if self.temp > self.min_temp:

            self.temp *= self.alpha

            return True

        return False

    def anneal(self):

        while self.temperature_schedule():

            neighbour = self.generate_neighbour()

            if self.accept(neighbour) > np.random.rand():

                self.route = neighbour

                if self.get_cost(neighbour) < self.get_cost(self.best_route):

                    self.best_route = neighbour

                    print("New best (temp = {:.5f}):".format(self.temp), self.get_cost(self.best_route))

        return self.best_route + 1, self.get_cost(self.best_route)


# http://katrinaeg.com/simulated-annealing.html
# https://github.com/chncyhn/simulated-annealing-tsp/blob/master/anneal.py
# http://www.psychicorigami.com/2007/06/28/tackling-the-travelling-salesman-problem-simmulated-annealing/
# https://arxiv.org/pdf/cs/0001018.pdf (Adaptive Simulated Annealing)
# https://www.ingber.com/ASA-README.html
# https://www.sciencedirect.com/science/article/pii/S0304414901000825
# http://www.iue.tuwien.ac.at/phd/binder/node87.html
# http://toddwschneider.com/posts/traveling-salesman-with-simulated-annealing-r-and-shiny/
