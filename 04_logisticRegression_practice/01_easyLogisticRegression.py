import numpy as np
import matplotlib.pyplot as plt


def load_data_set(file_path):
    """
    加载dataset文件夹中的数据集
    :param file_path: 文件目录
    :return:
    """
    data_matrix = []
    label_list = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip().split()
            data_matrix.append([float(line[0]), float(line[1]), 1.0])
            label_list.append(int(line[2]))
    return np.array(data_matrix), np.array(label_list)


def sigmoid(values):
    """
    sigmoid函数
    :param values: 列表类型 分别计算出值封装成列表返回
    :return: 1.0 / (1.0 + np.exp(-values))
    """
    return 1.0 / (1.0 + np.exp(-values))


def gradAscent(data_matrix, label_matrix, alpha=0.001, n=500):
    """
    开始使用梯度上升训练weights
    :param data_matrix: np.array()类型 数据集矩阵
    :param label_matrix: np.array()类型 标签矩阵
    :param alpha: 学习率
    :param n: 训练次数
    :return: weights
    """
    # label_matrix转为列向量
    label_matrix = np.mat(label_matrix).transpose()
    # 初始化weights = [3,1] = 1
    weights = np.ones(shape=(len(data_matrix[0]), 1))
    for i in range(n):
        # [n,1] = [n,m] * [m,1] 矩阵乘法
        # 要把data_matrix从np.array 转为np.mat才能相乘
        values = np.mat(data_matrix) * weights
        # 激活函数后得到类别
        activated_values = sigmoid(values)
        # 计算误差 (以2元函数举例)
        error_list = label_matrix - activated_values
        print(activated_values,error_list)
        # 调整权重weights
        # 因为偏移量error_list是(5,1)所以要把data_matrix转为(3,5)，最终得到每个权重需要调整的量
        weights = weights + alpha * data_matrix.transpose() * error_list
    return np.array(weights)


def paint(data_matrix, label_matrix, weights):
    """
    画出数据集和拟合直线
    :param data_matrix:
    :param label_matrix:
    :param weights:
    :return:
    """
    point_0 = []
    point_1 = []

    plt.figure(figsize=(5, 5), dpi=200)
    # 遍历把两类点分离出来
    for i in range(len(label_matrix)):
        if label_matrix[i] == 1:
            point_1.append(data_matrix[i])
        else:
            point_0.append(data_matrix[i])
    point_0 = np.array(point_0)
    point_1 = np.array(point_1)
    # 把离散点画出来
    plt.scatter(x=point_0[:, 0], y=point_0[:, 1], c="red")
    plt.scatter(x=point_1[:, 0], y=point_1[:, 1], c="blue")

    # 画拟合曲线
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[2] - weights[0] * x) / weights[1]
    plt.plot(x, y)

    plt.show()


if __name__ == '__main__':
    # 1' 加载数据集
    data_matrix, label_matrix = load_data_set("./dataset/01_set.txt")
    # 2' 梯度上升训练weights参数
    weights = gradAscent(data_matrix, label_matrix, n=5000)
    # print(weights)
    # 3' 把数据集和拟合直线画出来
    paint(data_matrix, label_matrix, weights)
