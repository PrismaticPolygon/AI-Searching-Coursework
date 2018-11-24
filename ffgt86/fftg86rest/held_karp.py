from fftg86rest.main import load_search_file, load_tour_file
import numpy as np
import math

filename = "AISearchtestcase.txt"

U, tour = load_tour_file(filename)
size, distance_matrix = load_search_file(filename)
iteration_factor = 0.015
max_changes = 100
t_small = 0.001
alpha = 2
beta = 0.5
num_iterations = int(iteration_factor * size)
h_star = -math.inf
d_T = size - 1  #

print("U: ", U)
print("Tour: ", tour)
print(distance_matrix)

v1 = np.random.choice(size)
e, f = None, None


for i, current in enumerate(tour):

    previous = tour[-1] if i == 0 else tour[i - 1]
    next = tour[0] if i == len(tour) - 1 else tour[i + 1]

    if current == v1:

        e = (previous, current)
        f = (current, next)

        break

c_e = distance_matrix[e[0], e[1]]
c_f = distance_matrix[f[0], f[1]]

A = sum(np.sort(distance_matrix[v1])[1:3])

# A set of edges consisting of two edges incident with node v1 plus a spanning tree of G \ v1 is a 1-tree.
# Optimal? No.

# Using this information, we update
# We can use our existing tours as starting points.


t_k = 1

# So we start with the y's off-kilter. Iterate through, using each vertex of the input tour as v1, then find
# the OPTIMUM minimum spanning tree, and the corresponding bound.
# And I can just use Prim's algorithm, right?
# With respect to each costs.

def generate_S():

    S = []

    for i in range(size):

        for j in range(i, size):

            if i is not j:
                S.append((i, j))

    return sorted(S, key=lambda n: distance_matrix[n[0], n[1]])

def cyclic(g):

    path = set()

    def visit(vertex):

        path.add(vertex)

        # It's not like we can have a key. 

        for neighbour in g.get(vertex, ()):

            if neighbour in path or visit(neighbour):

                return True

        path.remove(vertex)

        return False

    return any(visit(v) for v in g)




def get_optimum_subtree():

    S = generate_S()

    def contains_cycle(graph, a):

        # Or I include the graph and see if it causes a cycle. Is that easy?

        # for edge in graph:
        #
        #     if edge[0] == a[0]



        # We need to know if there's another way to get from a[0] to a[1].

        vertices = []

        for edge in graph:

            vertices.append(edge[0])
            vertices.append(edge[1])

        vertices = set(vertices)

        #print(vertices, a)

        if a[0] in vertices and a[1] in vertices:

            return True

        return False

        # Change this.

    print(S)


    MST = [S[0]]

    print(MST)

    i = 1

    # It's a start.

    while len(MST) < (size - 1):

        edge = S[i]

        print("Edge: ", edge)

        # Must be a bug in my contains cycle code. And there it is!

        if contains_cycle(MST, edge) is False:

            MST.append(edge)

            print(MST)

        # But I increment i.

        i += 1

    print(MST)

# for v in np.random.choice(size, size, replace=False):

    # Sort all edges in non-decreasing order of their weight.

    # e, f = None, None
    #
    # print("v1: ", v1)
    #
    # for i, current in enumerate(tour):
    #
    #     previous = tour[-1] if i == 0 else tour[i - 1]
    #     next = tour[0] if i == len(tour) else tour[i + 1]
    #
    #     if current == v1:
    #
    #         e = (previous, current)
    #         f = (current, next)
    #
    #         break
    #
    # c_e = distance_matrix[e[0], e[1]]
    # c_f = distance_matrix[f[0], f[1]]

    A = sum(np.sort(distance_matrix[v1])[1:3])

    # But it's a spanning tree in G \ v

get_optimum_subtree()

y = [0 for x in range(size)]

for i in range(max_changes):

    for k in range(num_iterations):

        # So here I just calculate that optimum 1-tree

        # Let T be the optimum 1 tree with respect to edge costs, and let H be the corresponding Held-Karp Bound.
        # That's the kicker.
        # It will be a tour, because the graph is connected. No it won't: we can't visit a vertex more than once.
        # That will be... interesting to compute. And how will I represent a tree? A list of expanded tuples?

        T = []
        h = 0

        if h > h_star:

            h_star = h

        if T is not None: # If T is a tour, stop and return.

            print("T is a tour! STOP!")

        t_k = (alpha * (U - h)) / (size * (2 - d_T) ** 2)

        if t_k < t_small:

            print("t_k < t_small! STOP!")

        y = [x + t_k * (3 - size) for x in y]

    alpha = beta * alpha

# From a given tour, remove two edges incident from a given node. Compute a minimum-cost spanning tree;
# this will give us a lower bound on the cost of the path.