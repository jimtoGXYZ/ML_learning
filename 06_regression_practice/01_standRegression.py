import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import linalg


def load_data(file_path):
    """
    加载数据集
    :param file_path: 路径
    :return:
    """
    df = pd.read_csv(file_path)
    # [n,m-1]
    data_matrix = np.mat(df.iloc[:, :-1])
    # [n,1]
    label_matrix = np.mat(df.iloc[:, -1]).T
    # print(np.shape(data_matrix), "\n")
    # print(np.shape(label_matrix), "\n")
    # print(data_matrix, "\n")
    # print(label_matrix)

    return data_matrix, label_matrix


def standRegress(data_matrix, label_matrix):
    """
    标准回归
    :param data_matrix:
    :param label_matrix:
    :return:
    """
    # 准备xTx 即x的转置乘x阵，用于求行列式从而判断是否可逆 [m-1,m-1] = [m-1,n] * [n,m-1]
    xTx = data_matrix.T * data_matrix
    # print(xTx)
    # 判断行列式
    if linalg.det(xTx) == 0:
        print("该矩阵不可逆")
        return

    # 若矩阵可逆 直接使用最小二乘法(即公式法)来算系数矩阵w w = [m-1,m-1] * [m-1,n] * [n,1] = [m-1,1]
    w = xTx.I * data_matrix.T * label_matrix
    print("W阵：", w)
    return w


def my_paint(w, data_matrix, label_matrix):
    """
    画图
    :param w:
    :param data_matrix:
    :param label_matrix:
    :return:
    """
    fig = plt.figure(figsize=(10, 10), dpi=200)
    # print(data_matrix[:, 1])
    # print(data_matrix[:, 1].flatten())
    # print(label_matrix.T)
    # print(label_matrix.T.flatten())
    # print(label_matrix.T.flatten().A[0])
    plt.scatter(x=data_matrix[:, 1].flatten().A[0], y=label_matrix.T.flatten().A[0])

    data_matrix_copied = data_matrix.copy()
    data_matrix_copied.sort(0)
    print(data_matrix_copied)
    y_predict = data_matrix_copied * w
    plt.plot(data_matrix_copied[:, 1], y_predict)

    plt.show()


def cal_rss(y_predict, y_true):
    sum = 0.0
    for i in range(len(y_predict)):
        sum += (y_predict[i] - y_true[i]) ** 2

    return sum[0][0]


def cal_mse(y_predict, y_true):
    sum = 0.0
    for i in range(len(y_predict)):
        sum += (y_predict[i] - y_true[i]) ** 2
    return sum / float(len(y_predict))


if __name__ == '__main__':
    file_path = "./dataSet/3_1.csv"
    # 加载数据
    data_matrix, label_matrix = load_data(file_path)

    # 若第0列不是1 的话画出来的回归直线不完全直 因为还有一个x0来影响y 所以为了更好地效果就把第0列赋值1
    # 可以反复注释下面这行代码看差别
    data_matrix[:, 0] = 1

    print(data_matrix)
    # 标准回归
    w = standRegress(data_matrix, label_matrix)
    # 画图
    my_paint(w, data_matrix, label_matrix)
    y_predict = data_matrix * w
    # 计算RSS
    rss = cal_rss(y_predict, label_matrix)
    print("rss:", rss)
    # 计算MSE
    mse = cal_mse(y_predict, label_matrix)
    print("mse:", mse)
