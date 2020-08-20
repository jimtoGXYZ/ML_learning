import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def draw_the_pic():
    file_path = "../dataSet/2.csv"
    df = pd.read_csv(file_path)
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = plt.gca(projection='3d')
    ax.plot(df["x0"], df["x1"], df["y"])
    plt.show()


def draw_the_pic2():
    file_path = "../dataSet/2.csv"
    df = pd.read_csv(file_path)
    fig = plt.figure(figsize=(10, 10), dpi=200)
    plt.scatter(x=df["x0"], y=df["y"])
    plt.show()

def draw_the_pic3():
    file_path = "../dataSet/2.csv"
    df = pd.read_csv(file_path)
    fig = plt.figure(figsize=(10, 10), dpi=200)
    plt.scatter(x=df["x1"], y=df["y"])
    plt.show()



if __name__ == '__main__':
    draw_the_pic()
    draw_the_pic2()
    draw_the_pic3()
