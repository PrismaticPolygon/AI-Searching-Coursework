import numpy as np
import re
import os
from ffgt86.fftg86rest.simulated_annealing import SimulatedAnnealing
from ffgt86.fftg86rest.genetic_algorithm import GeneticAlgorithm


def get_files():

    for filename in reversed(os.listdir("../../cityfiles")):

            yield load_search_file(filename)


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

    return filename, matrix, matrix.shape[0]


def load_tour_file(filename, algorithm):

    with open("../Tourfile" + algorithm + "/tour" + filename, 'r') as file:

        length = 0
        tour = []

        for i, line in enumerate(file):

            text = line.replace("\n", '')

            if i == 2:

                length = int(text[9:-1])

            if i == 3:

                tour = [int(x) - 1 for x in text.split(",")]

        return filename, length, tour


def write_file(filename, algorithm, tour, length):

    tour_filename = "../Tourfile" + algorithm + "/tour" + filename

    def write():

        print("Writing to {}: ".format(filename), algorithm, tour, length)

        with open(tour_filename, "w") as f:

            try:

                f.write("NAME = " + filename[3:-4] + ",\n")
                f.write("TOURSIZE = " + str(tour.size) + ",\n")
                f.write("LENGTH = " + str(length) + ",\n")
                f.write(",".join(map(str, tour)))

            finally:

                f.close()

    try:

        filename, old_length, tour = load_tour_file(filename, algorithm)

        if old_length > length:

            write()

    except IOError:

        write()


def load_iterative_files_sa():

    for file in (os.listdir("iterative_sa/")):

        print("\n" + file + "\n")

        tours = {}

        for iteration in os.listdir("iterative_sa/" + file):

            with open("iterative_sa/" + file + "/" + iteration, "r") as f:

                lines = f.readlines()

                length = int(lines[2][9:-2])
                alpha = float(lines[4][7:-2])
                min_temp = float(lines[5][10:-2])
                best_time = float(lines[6][11:-2])

                if length not in tours:

                    tours[length] = [[alpha, min_temp, best_time]]

                else:

                    tours[length].append([alpha, min_temp, best_time])

        best_length = min(tours)
        best_tours = []

        for key, value in tours.items():

            if key <= int(1.02 * best_length):

                best_tours += value

        mean_alpha = sum([x[0] for x in best_tours]) / len(best_tours)
        mean_min_temp = sum([x[1] for x in best_tours]) / len(best_tours)
        mean_best_time = sum([x[2] for x in best_tours]) / len(best_tours)

        print("Best tours: ", best_tours)
        print("Mean alpha: ", mean_alpha)
        print("Mean min temp: ", mean_min_temp)
        print("Mean best time: ", mean_best_time)

        best_ever = min(tours[best_length], key=lambda x: x[2])

        print(str(best_length) + ": ", best_ever)


def load_iterative_files_ga():

    for file in (os.listdir("iterative_ga/new/")):

        if file == "new":

            continue

        print("\n" + file + "\n") # Cause the rest are in use, I'm guessing. I wonder if I'll get an I/O exception trying to copy them

        tours = {}

        for iteration in os.listdir("iterative_ga/new/" + file):

            with open("iterative_ga/new/" + file + "/" + iteration, "r") as f:

                lines = f.readlines()

                length = int(lines[2][9:-2])
                k = int(lines[4][3:-2])
                crossover_probability = float(lines[5][4:-2])
                mutation_probability = float(lines[6][4:-2])
                generation = int(lines[7][12:-2])

                if length not in tours:

                    tours[length] = [[k, crossover_probability, mutation_probability, generation]]

                else:

                    tours[length].append([k, crossover_probability, mutation_probability, generation])

        best_length = min(tours)
        best_tours = []

        for key, value in tours.items():

            if key <= int(1.00 * best_length):

                best_tours += value

        mean_k = sum([x[0] for x in best_tours]) / len(best_tours)
        mean_pc = sum([x[1] for x in best_tours]) / len(best_tours)
        mean_pm = sum([x[2] for x in best_tours]) / len(best_tours)
        mean_g = sum([x[3] for x in best_tours]) / len(best_tours)

        print("Best tours: ", best_tours)
        print("Mean tournament size: ", mean_k)
        print("Mean crossover probability: ", mean_pc)
        print("Mean mutation probability: ", mean_pm)
        print("Mean generation: ", mean_g)

        best_ever = min(tours[best_length], key=lambda x: x[3])

        print(str(min(tours)) + ": ", best_ever)


def write_iterative_file(filename, ga, tour, length):

    directory = "iterative_ga/new/" + filename[:-4] + "/"

    if not os.path.isdir(directory):

        os.makedirs(directory)

    tour_filename = directory + filename + " ({}, {}, {})".format(ga.tournament_size, ga.crossover_probability, ga.mutation_probability)

    with open(tour_filename, "w") as f:

        try:

            f.write("NAME = " + filename[3:-4] + ",\n")
            f.write("TOURSIZE = " + str(tour.size) + ",\n")
            f.write("LENGTH = " + str(length) + ",\n")
            f.write(",".join(map(str, tour)) + ",\n")
            f.write("k: " + str(ga.tournament_size) + ",\n")
            f.write("P_c: " + str(ga.crossover_probability) + ",\n")
            f.write("P_m: " + str(ga.mutation_probability) + ",\n")
            f.write("GENERATION: " + str(ga.best_generation) + ",\n")

        finally:

            f.close()


def write_iterative_file_sa(filename, sa, tour, length):

    directory = "iterative_sa/" + filename[:-4] + "/"

    tour_filename = directory + filename + " ({}, {})".format(sa.alpha, sa.min_temp)

    if not os.path.isdir(directory):

        os.makedirs(directory)

    with open(tour_filename, "w") as f:

        try:

            f.write("NAME = " + filename[3:-4] + ",\n")
            f.write("TOURSIZE = " + str(tour.size) + ",\n")
            f.write("LENGTH = " + str(length) + ",\n")
            f.write(",".join(map(str, tour)) + ",\n")
            f.write("alpha: " + str(sa.alpha) + ",\n")
            f.write("min_temp: " + str(sa.min_temp) + ",\n")
            f.write("best_time: " + str(sa.best_route_time) + ",\n")

        finally:

            f.close()


def run_ga_iteratively(filename, distance_matrix, length):

    for tournament_size in range(1, 10):

        for crossover_probability in [(x / 10) for x in range(10)]:

            for mutation_probability in [(x / 10) for x in range(10)]:

                print("\nGenetic algorithm ({}, {}, {})\n".format(tournament_size, crossover_probability, mutation_probability))

                ga = GeneticAlgorithm(distance_matrix, length, mutation_probability, crossover_probability, tournament_size)
                ga_tour, ga_cost = ga.evolve()
                write_iterative_file(filename, ga, ga_tour, ga_cost)


def run_sa_iteratively(filename, distance_matrix, length):

    for alpha in [(1 - x / 100) for x in range(0, 20, 2)]:

        for min_temp in [x for x in range(1, 10)]:

            print("\nSimulated annealing ({}, {})\n".format(alpha, min_temp))

            sa = SimulatedAnnealing(distance_matrix, length, alpha, min_temp)
            sa_tour, sa_cost = sa.anneal()
            write_iterative_file_sa(filename, sa, sa_tour, sa_cost)


print("\nRunning algorithms...")

for filename, distance_matrix, length in get_files():

    print(filename)

    print("\nGenetic algorithm\n")

    ga = GeneticAlgorithm(distance_matrix, length)
    ga_tour, ga_cost = ga.evolve()
    write_file(filename, "A", ga_tour, ga_cost)

    print("\nSimulated annealing\n")

    sa = SimulatedAnnealing(distance_matrix, length)
    sa_tour, sa_cost = sa.anneal()
    write_file(filename, "B", sa_tour, sa_cost)
