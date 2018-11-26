import numpy as np
import time


class SimulatedAnnealing:

    def __init__(self, distance_matrix, length, min_temp=1, alpha=1, runtime=60):

        self.distance_matrix = distance_matrix
        self.length = length
        self.temp = self.min_temp = min_temp
        self.alpha = alpha
        self.runtime = runtime
        self.r = 0
        self.tabu = []
        self.start_time = time.time()

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
        self.tabu.append(route.tolist())

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

            self.r = 0
            return True

        elif np.exp((current_cost - new_cost) / self.temp) > np.random.rand():

            self.r += 1
            return True

        return False

    def adaptive_temperature_schedule(self):

        if time.time() - self.start_time < self.runtime:

            self.temp = self.min_temp + self.alpha * np.log(1 + self.r)

            return True

        return False

    def anneal(self):

        while self.adaptive_temperature_schedule():

            neighbour = self.generate_neighbour()

            if self.accept(neighbour) and neighbour.tolist() not in self.tabu:

                self.route = neighbour
                self.tabu.append(neighbour.tolist())

                if self.get_cost(self.route) < self.get_cost(self.best_route):

                    self.best_route = self.route

                    print("New best (temp = {:.5f}):".format(self.temp), self.get_cost(self.best_route))

        return self.best_route + 1, self.get_cost(self.best_route)
