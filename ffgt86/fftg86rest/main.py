import numpy as np
import re
import os
from ffgt86.fftg86rest.simulated_annealing import SimulatedAnnealing
from ffgt86.fftg86rest.genetic_algorithm import GeneticAlgorithm


def get_files():

    for file in reversed(os.listdir("../../cityfiles")):

        if file != "AISearchtestcase.txt":

            yield file, load_search_file(file)


def load_search_file(filename):

    with open("../../cityfiles/" + filename, "r") as file:

        data = file.read().replace("\n", '').split(",")

        size = int(data[1][7:])
        matrix = np.zeros((size, size), dtype=int)
        j, i = 0, 1

        for distance in data[2:]:

            if i == size:

                j += 1
                size -= 1
                i -= size

            try:

                matrix[j, i + j] = matrix[i + j, j] = re.sub(r'[^0-9]+', "", distance)

            except Exception as e:

                print("Error building matrix: ", e)

            i += 1

    return matrix, matrix.shape[0]


def load_tour_file(filename, algorithm="A"):

    with open("../Tourfile" + algorithm + "/tour" + filename, 'r') as file:

        length, tour = 0, []

        for i, line in enumerate(file):

            text = line.replace("\n", '')

            if i == 1:

                length = int(text[11:-1])

            if i == 3:

                tour = [int(x) - 1 for x in text.split(",")]

        return length, tour


def write_file(filename, algorithm, tour, length):

    tour_filename = "../Tourfile" + algorithm + "/tour" + filename

    def write():

        print("Writing to {}: ".format(filename), algorithm, tour, length)

        with open(tour_filename, "w") as f:

            try:

                f.write("NAME = " + filename[:-4] + ",\n")
                f.write("TOURSIZE = " + str(tour.size) + ",\n")
                f.write("LENGTH = " + str(length) + ",\n")
                f.write(",".join(map(str, tour)))

            finally:

                f.close()

    try:

        with open(tour_filename, "r") as f:

            old_length = int(f.readlines()[2][9:-2])

            if old_length > length:

                write()

    except IOError:

        write()


print("\nRunning algorithms...\n")

for filename, (distance_matrix, length) in get_files():

    print(filename)

    print("\nGenetic algorithm\n")

    ga = GeneticAlgorithm(distance_matrix, length)
    ga_tour, ga_cost = ga.evolve()
    write_file(filename, "A", ga_tour, ga_cost)

    print("\nSimulated annealing\n")

    sa = SimulatedAnnealing(distance_matrix, length)
    sa_tour, sa_cost = sa.anneal()
    write_file(filename, "B", sa_tour, sa_cost)

    print()
