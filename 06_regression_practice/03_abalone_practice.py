import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg


def load_data(file_path):
    """
    针对鲍鱼数据集做数据加载 需要把标称数据转为二值型数据
    :param file_path: 路径
    :return:
    """
    df = pd.read_csv(file_path)
    # 取前一百条数据训练 否则太久了
    df = df[:100]
    # 把sex改为 F I M 三列
    sex_array = df["Sex"]
    data_transformed = rearrange_nominal_data(sex_array)
    df["F"] = data_transformed["F"]
    df["I"] = data_transformed["I"]
    df["M"] = data_transformed["M"]
    # 删除原来的Sex
    df.drop(labels="Sex", axis=1, inplace=True)

    # 把Rings维持在最后一列
    tmp_array = df["Rings"]
    df.drop(labels="Rings", axis=1, inplace=True)
    df["Rings"] = tmp_array

    data_matrix = np.mat(df.iloc[:, :-1])
    # [n,1]
    label_matrix = np.mat(df.iloc[:, -1]).T

    return data_matrix, label_matrix


def lwlr(testPoint, data_matrix, label_matrix, k=0.5):
    weights = np.mat(np.eye(len(data_matrix)))
    for i in range(len(data_matrix)):
        # 计算特征距离
        distance_matrix = testPoint - data_matrix[i, :]
        print("testPoint:", testPoint)
        print("data_i:", data_matrix[i, :])
        print("distance:", distance_matrix)
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
    测试局部加权线性回归
    :param data_matrxi:
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


def my_draw(data_matrix, label_matrix, y_predict, i):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    sorted_index = data_matrix[:, 1].argsort(0)
    x_sorted = data_matrix[sorted_index][:, 0, :]
    # print(x_sorted)

    plt.plot(x_sorted[:, 1], y_predict[sorted_index])
    plt.scatter(x=data_matrix[:, 1].flatten().A[0], y=label_matrix.T.flatten().A[0], c='red')
    plt.savefig("./dataSet/abalone_%d.png" % i)


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


def rearrange_nominal_data(feature_array):
    data_transformed = pd.get_dummies(feature_array)
    return data_transformed


if __name__ == '__main__':
    # 加载数据
    file_path = "./dataSet/abalone.csv"
    data_matrix, label_matrix = load_data(file_path)
    print(data_matrix)
    for i in [1, 5, 10]:
        print("第%d轮" % i)
        # 得到估计值y_predict
        y_predict = lwlr_predict(data_matrix, data_matrix, label_matrix, i)
        # 画图
        my_draw(data_matrix, label_matrix, y_predict, i)
        # 计算rss mse
        rss = cal_rss(y_predict, label_matrix)
        mse = cal_mse(y_predict, label_matrix)
        print("第%d个核的rss:%f" % (i, rss))
        print("第%d个核的mse:%f" % (i, mse))
