import pandas

tour_file = open("AISearchtestcase.txt", "r")
name=""

with open("AISearchtestcase.txt", "r") as file:

    for i, line in enumerate(file):

        if i == 0:

            name = line[7:-2]

        print(i, line)

print(name)