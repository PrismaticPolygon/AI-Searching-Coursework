from fftg86rest.main import load_search_file
import numpy as np
import random

size, distance_matrix = load_search_file("AISearchtestcase.txt")

# Let's generate T as a list of tuples. To keep it in line with S!

#T = np.random.choice(size, size, replace=False)
# best improvement made so far.
# Choose any node ti and let xi be one of the edges of T adjacent to ti.

def generate_S():

    S = []

    for i in range(size):

        for j in range(i, size):

            if i is not j:
                S.append((i, j))

    return S


def generate_edge():

    x = np.random.randint(0, size)

    while True:

        y = np.random.randint(0, size)

        if x is not y:

            return x, y


def generate_edges(t):

    e = []

    for j in range(size):

        if j != t:

            e.append((t, j))

    return e


def gain(x, y):

    return abs(distance_matrix[x[0], x[1]]) - abs(distance_matrix[y[0], y[1]])


def generate_T():

    T = [(np.random.randint(0, size), np.random.randint(0, size))]

    for i in range(1, size):

        previous = T[-1][1]
        used = [i[0] for i in T]
        second_used = [i[1] for i in T]

        for j in np.random.choice(size, size, replace=False):

            if j is not previous and j not in used and j not in second_used and (previous, j) not in T:

                T.append((previous, j))

                break

    T.append((T[-1][1], T[0][0]))

    return T


def get_adjacent_edge(x):

    edges = generate_edges(x[1])

    for edge in edges:

        if edge not in T and gain(x, edge) > 0:

            return edge

    return None

def get_nearest_node(node):

    min = None

    for j in distance_matrix[node]:

        if min is None or min > j:

            min = j

    return min


def choose_next_x(y):

    return y[1], np.random.randint(0, size)

# S is the set of all edges.
# S - T is the set of all edges NOT in T.
# We're attempting to find two sets of links X and Y such that, if the links in X are deleted or 'broken'
# and replaced by the links in Y, the result is a tour of lower cost.
# xi and yi share an endpoint; so do yi and x(i + 1)

# What a fucking rabbit hole this is.
#   The resulting configuration is a tour. What the fuck does that mean? What fucking configuration?
S = generate_S()
T = generate_T()
G_star = []

x = []
y = []
# We need a starting x to work from, right?
x_0 = random.choice(T)
y_0 = get_adjacent_edge(x_0)

# So what do we choose k to be? Didn't they say that it shouldn't be

print(T)

while True:

    i = 0

    x.append(np.random.choice(T))
    y.append(get_adjacent_edge(x[-1]))

    if y[-1] is None:

        print("Panic! Got to step 6(d), the first application of the gain criterion. ")
        break

    i += 1

    # Each xi is an edge in the current tour.
    # Each yi is NOT in the current tour
    # All are unique.
    # The last yk returns to the starting point t1.

    x.append(choose_next_x(y[-1]))

    #yi is some available link at the endpoint t2i shared with xi, chosen with nearest preferentially.

    # xi is chosen such that, if t2i is joined to ti, the resulting configuration is a tour.
    # So, for any given y-1, xi is uniquely determined, for any i >= 2. The choice of yi-1 ensures this is possible.

    # In order the ensure the feasibility criterion of a can be satisified at i + 1, the ui must permit the breaking
    # of an xi+1

    # It's because we're replacing vertices! So to complete the tour, x_i need to connect to the index past
    # the current length of x.

    # How on earth do I check whether it's a tour. Oh, it's simple. Right?

    # The resulting configuration is a tour...

    break

# x_1 corresponds to choosing a random edge from T
#
# x_1 = None
#
# for edge in T:
#
#     if edge[0] == t_1:
#
#         x_1 = (t_1, edge[1])
x1 = random.choice(T)
print("x1 =", x1)

# x_1 is the first replacement edge.

i = 1
y1 = None

edges = generate_edges(x1[1])
print(edges)
# How on earth did it get (2, 2)?

for edge in generate_edges(x1[1]):

    if gain(x1, edge) > 0:

        y1 = edge
        break

if y1 is None:

    print("Panic! Got to step 6(d), the first application of the gain criterion. ")

# What's y1? Ah, and edge connecting t_2 to t_3

print("y_1 =", y1)

i += 1

x2 = (y1[1], np.random.randint(0, size))

print("x2 =", x2)

y2 = None

for edge in generate_edges(x2[1]):

    if gain(x2, edge) > 0:

        y2 = edge
        break

if y2 is None:

    print("Panic! Go to step 5")

print("y2 =", y2)

if x2 == y1:

    print("Not disjoint! Someone do something")

g1_star = gain(x1, y1) + gain(x2, y2)

if g1_star < 0:

    print("Gain is not positive! This is bad")

# Monotone non-decreasing: a function between ordered sets that preserves the order.

# In order to ensure that the feasibility criterion of (a) can be satisifed at i +1, the yi chosen
# must permit the breaking of an xi+1. What? We don't need to worry about this: the graph is complete! Lol.

#

# G_star is a list of sets, I guess.
# Wait a fucking minute.
# G* must be the sum of gains. 

y2_star = (x2[1], x1[0])
g2_star = gain(y2_star, x2)

# If the sum of gains - excluding the most recent plus the newly generated gain is greater than the sum of gains,
# switch it up.
# It's the Kleene star: G* is the union of all Gs. Which just means the sum of gains, right?


# If the previous value of G + g2_star is greater than G*.
# Ah. If G, minus the previously added value, plus g2_star is


# There's a bunch of criteria for choosing the next x; we should make it a function.

# The sum of the gains thus far must be greater than 0.

# xi cannot be a link previously joined, and yi cannot be a link previously broken. Broken?
# I.e. xi cannot be a yj, less than.

# yi is some available link at endpoint t2i (currently t4 = x2[1]) shared with x2.
# Repeat the above process, right?


# This corresponds to choosing another random edge, right?

# So we're choosing x_2, connecting t_3 and t_4

# previous index is 1, connecting t_2. Ah. We already have t_2.

# Time to index? Let's do it for one iteration first .

# Choose xi, which currently joins t2i-1 to t2i. Such that the resulting configuration is a tour. Easy!
# yi is some available link at the endpoint t2i, shared with xi.

    # If no such y1 exists (how on earth do I check that? I'd have to iterate through every possibility...
    # and, to be fair, that's not that hard.  We can make it a generator later.




# How do I choose, though? Iteratively? Or just randomly?

# From the other endpoint t_2 of x_1, choose.








S = []

for i in range(size):

    for j in range(i, size):

        if i is not j:

            S.append((i, j))



# If a sequence of numbers has a positive sum, there is a cyclic permutation of these numbers such that
# every partial sum is positive.
# As we are looking for sequences of gi that always have a positive sum, we need only consider sequences of gains
# whose partial sum is always positive. This is CRUCIAL.

# For now: let's do it as a list.

# S is the set of edges.
# T is an n-subset of S that forms a tour.

# S is the set of all links, then. Which means T should be something else; a list of tuples of links.

# Transform T into T' by identifying sequentially the k pairs of links to be exchanged between T and S - T.

#Select xi and yi as the most-out-of place pair at the ith step. xi and yi are chosen to maximise the improvement
# when x1 through xi are exchanged. xi is chosen from T - {x}, yi from S - T - {y}.

# If it appears that no more gain can be made, continue, else set i = i + 1
# If the best improvement is found for i = k, exchange x1 through xk with y1 through yk to give a new T, then go to
# step 2; else go to step 4 (repeat from step 1 as desired).

# Selection rule that tells us which pair is currently most out of place.
# Function that returns the total profit from a proposed set of exchanges. Additive, so no need to stop immediately when
# gi is negative, online when the sum of profits is negative for all ks of its. Helps evading local minima.

# Each exchange must leave us in a feasible set (easy; the graphs are complete!).

# A stopping rule to tell us that there can be no further profit in looking for elements to exchange, or that we've
# reached the point of diminishing returns.

# Finally, that the sets x and y be disjoint. (no elements in common). Once an element is moved one way, it is not
# returned during this iteration.

# After examining some sequence x and y of proposed exchanges, actual value of k that defines the sets to exchange is
# the one that maximises the sum profit. Once g is always zero or negative, stop.

# Start, again, with random, uniformly distributed starting solutions.

# T is not optimal because there are k elements x1 through xk that are 'out of place'.
# To make optimal, they should be replaced with k elements y1 through yk of S-T.
# Problem is then to identify k, x, and y.

# Try to find k, x1 through xk, and y1 through yk as best we can, element by element.
# Try to identify x1 and y1 as the most out-of-place pair, set them aside, and find x2 and y2,
# the most out-of-place pair in the remaining sets.
