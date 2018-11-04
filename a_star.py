from main import load_file
import numpy as np

# Guided by a heuristic function h(n), the estimate distance from the node n to the goal node.
# Each loop iteration, A* selects the path that minimises f(n) = g(n) + h(n),
# where g(n) is the cost of the path from the start node to n


# So cameFrom is a dict. A dict of what, though?
# current is the node we're currently on


def reconstruct_path(came_from, current):

    total_path = [current]

    while current in came_from.keys():

        current = came_from[current]
        total_path.append(current)

    return total_path


def a_star(start, goal):

    closed_set = []  # Set of nodes already evaluated
    open_set = [start]  # Set of currently discovered nodes that are not evaluated yet

    # For each node, which node is can most efficiently be reached from.
    came_from = {}

    # For each node, the cost of getting from the start node to that node
    g_score = dict()
    g_score[start] = 0

    # For each node, the total cost of getting from the start node to that node. Partly known, partly heuristic
    f_score = dict()
    f_score[start] = heuristic_cost_estimate(start, goal)

    # Use minimum spanning tree as a heuristic.
    # Initial state: agent in the start city and has not visited any other city
    # Successor function: generates all cities that have not yet been visited
    # Edge-cost: distance between the cities, used to calculate g(n)
    # h(n): distance to the nearest unvisited city from the current city + estimated distance to travel to all unvisited cities + nearest distance from an unvisited city to the start city


    # A spanning tree is a subgraph tha contains all vertices

    while len(open_set) != 0:

        print(open_set)

        current = start

        # Get the node with the minimum f_score value that is in open_set.
        for node in open_set:

            if f_score[node] < f_score[current]:

                current = node

        print("Current node: " + str(current))

        if current == goal:

            return reconstruct_path(came_from, current)

        open_set.remove(current)
        closed_set.append(current)

        for neighbour in cities:

            print("Checking neighbour: " + str(neighbour))

            if neighbour in closed_set or distance_matrix[current][neighbour] == 0:

                continue

            tentative_g_score = g_score[current] + distance_matrix[current][neighbour]

            print("Tentative g score: " + str(tentative_g_score))

            if neighbour not in open_set:

                open_set.append(neighbour)

            elif tentative_g_score >= g_score[neighbour]:

                continue

            came_from[neighbour] = current
            g_score[neighbour] = tentative_g_score
            f_score[neighbour] = g_score[neighbour] + heuristic_cost_estimate(neighbour, goal)


def minimum_spanning_tree(visited_cities):

    unvisited_matrix = np.delete(distance_matrix, visited_cities)

    print(unvisited_matrix)

    # Only unvisited cities, though.

    # How do I even sort the edges, lol?

    print("Lol")
    # Sort edges of G in increasing order of cost.
    # Keep a subgraph S of G, initially empty
    # For each edge e in sorted order, if the endpoints of e are disconnected in S (i.e. do not form a cycle)
        # Add e to S
    # Return S




def heuristic_cost_estimate(start, goal):

    return 1


cities, distance_matrix = load_file("AISearchtestcase.txt")

# How do we choose the start node, though? I don't think that it was specified.
# Surely I can't just iterate through them all... I'll get my implementation working first, then
# figure out how to apply it to the TSP specifically.

#a_star(0, 7)