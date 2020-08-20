import random


def buil_data(n=500):
    for i in range(n):
        x0 = round(1.0 + random.uniform(-0.05, 0.05), 3)
        x1 = round(0.8 + random.uniform(-0.05, 0.06), 3)
        y = round((1.2 + random.uniform(-0.02, 0.03)) * x0 + (0.9 + random.uniform(-0.02, 0.03)) * x1, 3)
        with open("../dataSet/2.csv", "a") as f:
            f.write(str(x0) + "," + str(x1) + "," + str(y) + "\n")

if __name__ == '__main__':
    buil_data()