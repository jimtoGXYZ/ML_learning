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


def lwlr(testPoint, data_matrix, label_matrix, k=0.5):
    weights = np.mat(np.eye(len(data_matrix)))
    for i in range(len(data_matrix)):
        # 计算该实例与其他点的距离
        distance_matrix = testPoint - data_matrix[i, :]
        # print("testPoint:", testPoint)
        # print("data_i:", data_matrix[i, :])
        # print("distance:", distance_matrix)
        weights[i, i] = np.exp(distance_matrix * distance_matrix.T / (-2.0 * k ** 2))
    xTx = data_matrix.T * (weights * data_matrix)
    if linalg.det(xTx) == 0:
        print("行列式为0")
        return

    # print(np.shape(xTx))
    # print(np.shape(data_matrix.T))
    # print(np.shape(label_matrix))
    w = xTx.I * (data_matrix.T * (weights * label_matrix))
    return testPoint * w


def lwlr_predict(test_matrix, data_matrix, label_matrix, k=0.5):
    """
    调用局部加权线性回归
    通过遍历对每一个实例调用局部加权线性回归函数来得到对应预测值
    :param test_matrix:
    :param data_matrix:
    :param label_matrix:
    :param k:
    :return:
    """
    y_predict = np.zeros(len(data_matrix))
    # print(y_predict)
    for i in range(len(data_matrix)):
        y_predict[i] = lwlr(test_matrix[i], data_matrix, label_matrix, k)

    # 返回估计值
    return y_predict


def my_draw(data_matrix, label_matrix, y_predict):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    sorted_index = data_matrix[:, 1].argsort(0)
    x_sorted = data_matrix[sorted_index][:, 0, :]
    # print(x_sorted)

    plt.plot(x_sorted[:, 1], y_predict[sorted_index])
    plt.scatter(x=data_matrix[:, 1].flatten().A[0], y=label_matrix.T.flatten().A[0], c='red')
    plt.savefig("./dataSet/2.png")


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
    # 加载数据
    file_path = "./dataSet/2.csv"
    data_matrix, label_matrix = load_data(file_path)
    # 同理01.py
    data_matrix[:, 0] = 1.0
    # 得到估计值y_predict
    y_predict = lwlr_predict(data_matrix, data_matrix, label_matrix, 0.01)
    # 画图
    my_draw(data_matrix, label_matrix, y_predict)
    # 计算rss mse
    rss = cal_rss(y_predict, label_matrix)
    mse = cal_mse(y_predict, label_matrix)
    print("rss:", rss)
    print("mse:", mse)
