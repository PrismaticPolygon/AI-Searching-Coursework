import numpy as np
import re
import os


def get_files():

    for file in os.listdir("search_data"):

        yield load_file(file)


def load_file(filename):

    with open("search_data/" + filename, "r") as file:

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

    return matrix.shape[0], matrix


def write_file(filename, tour, length):

    try:

        f = open("tour" + filename, "w")

        try:

            f.write("NAME = " + filename[:-4] + ",\n")
            f.write("TOURSIZE = " + str(tour.size) + ",\n")
            f.write("LENGTH = " + str(length) + ",\n")
            f.write(",".join(map(str, tour)))

        finally:

            f.close()

    except IOError:

        print("Error writing file")
        pass





#TODO: catch bad file names