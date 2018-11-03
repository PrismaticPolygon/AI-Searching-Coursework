from main import load_file
import numpy as np
import math


size, distance_matrix = load_file("AISearchtestcase.txt")
temp = 1

# Hyperparameters
min_temp = 0.00001
alpha = 0.88
comparison_size = 450

# This is blinding fast, and very nice and simple.
# Then again, I have a great PC and it's a tiny graph.
# Time to improve this mofo.

# A solution class with an associated cost?

# Is that bad practice?


# class Solution:
#
#     def __init__(self):
#
#         self.route = np.random.choice(size, size, replace=False)
#
#         self.cost = distance_matrix[0, self.route[-1]]
#
#         for i in range(1, size):
#
#             self.cost += distance_matrix[i - 1, i]




def generate_solution(size):

    return np.random.choice(size, size, replace=False)


def calculate_acceptance_probability(old, new, temperature):

    return math.exp((new - old) / temperature)


def calculate_cost(solution):

    sum = distance_matrix[0, solution[-1]] # Travel from the end back to the start

    for i in range(1, len(solution)):

        sum += distance_matrix[i - 1, i]

    return sum


best_ever_solution = solution = generate_solution(size)
best_ever_cost = cost = calculate_cost(solution)

while temp > min_temp:

    i = 1

    while i < comparison_size:

        new_solution = generate_solution(size)
        new_cost = calculate_cost(new_solution)
        acceptance_probability = calculate_acceptance_probability(cost, new_cost, temp)

        if acceptance_probability > np.random.rand():

            solution = new_solution
            cost = new_cost

            if new_cost < best_ever_cost:

                best_ever_solution = new_solution
                best_ever_cost = new_cost

        i += 1

    temp *= alpha

print(solution, cost)

# http://katrinaeg.com/simulated-annealing.html
