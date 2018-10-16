import numpy as np
import re


def load_file(filename):

    with open("search_data/" + filename, "r") as file:

        data = file.read().replace("\n", '').split(",")

        size = int(data[1][7:])
        cities = np.zeros((size, size))
        j, i = 0, 1

        for distance in data[2:]:

            if i == size:

                j += 1
                size -= 1
                i -= size

            try:

                cities[j, i + j] = cities[i + j, j] = re.sub(r'[^0-9]+', "", distance)

            except Exception as e:

                print("Error building matrix: ", e)

            i += 1

    return cities