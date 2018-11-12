import numpy as np
import re
import os


def get_files():

    for file in reversed(os.listdir("search_data")):

        yield file, load_file(file)


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


def write_file(filename, algorithm, tour, length):

    def write():

        print("Writing to {}: ".format(filename), algorithm, tour, length)

        with open("tour_data/Tourfile" + algorithm + "/tour" + filename, "w") as f:

            try:

                f.write("NAME = " + filename[:-4] + ",\n")
                f.write("TOURSIZE = " + str(tour.size) + ",\n")
                f.write("LENGTH = " + str(length) + ",\n")
                f.write(",".join(map(str, tour)))

            finally:

                f.close()

    try:

        with open("tour_data/Tourfile" + algorithm + "/tour" + filename, "r") as f:

            old_length = int(f.readlines()[2][9:-2])

            if old_length > length:

                write()

    except IOError:

        write()


# 12: 56
# 17: 1200
# 21: 2321
# 26: 1576
# 42: 1177
# 48: 12203
# 58: 21245
# 175: 21763
# 535: 51341



#TODO: catch bad file names