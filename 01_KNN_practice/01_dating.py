import operator

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from numpy import tile


def read_data(file_path):
    """
    读取数据集
    :param file_path: 文件路径
    :return: dating_data
    """
    dating_data = pd.read_csv(file_path, delimiter="\t", names=["飞行公里数", "游戏时间比例", "雪糕消耗", "喜爱程度"])
    data = dating_data.iloc[:, :3]
    target = dating_data.iloc[:, 3]
    return data, target


def draw_raw_data_01(data):
    """
    画出海伦约会数据集中游戏时间与飞行时间分布散点图
    :param data: [1000,4] 矩阵
    :return: None
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    plt.figure(figsize=(20, 9), dpi=100)
    plt.xlabel("飞行公里数")
    plt.ylabel("游戏时间占比")
    plt.title("海伦约会数据集飞行公里数与游戏时间占比喜爱程度分布图")
    g1 = plt.scatter(x=data[data["喜爱程度"] == "largeDoses"]["飞行公里数"], y=data[data["喜爱程度"] == "largeDoses"]["游戏时间比例"],
                     c="red")
    g2 = plt.scatter(x=data[data["喜爱程度"] == "smallDoses"]["飞行公里数"], y=data[data["喜爱程度"] == "smallDoses"]["游戏时间比例"],
                     c="blue")
    g3 = plt.scatter(x=data[data["喜爱程度"] == "didntLike"]["飞行公里数"], y=data[data["喜爱程度"] == "didntLike"]["游戏时间比例"],
                     c="black")
    plt.legend(handles=[g1, g2, g3], labels=["largeDoses", "smallDoses", "didntLike"], prop={'size': 16})
    # plt.show()
    plt.savefig("./dataset/dating_pic/飞行时间与游戏占比分布图.png")


def draw_raw_data_02(data):
    """
    画出海伦约会数据集中飞行公里数与雪糕消耗分布散点图
    :param data: [1000,4] 矩阵
    :return: None
    """

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    plt.figure(figsize=(20, 9), dpi=100)
    plt.xlabel("飞行公里数")
    plt.ylabel("雪糕消耗")
    plt.title("海伦约会数据集飞行公里数与雪糕消耗占比喜爱程度分布图")
    g1 = plt.scatter(x=data[data["喜爱程度"] == "largeDoses"]["飞行公里数"], y=data[data["喜爱程度"] == "largeDoses"]["雪糕消耗"],
                     c="red")
    g2 = plt.scatter(x=data[data["喜爱程度"] == "smallDoses"]["飞行公里数"], y=data[data["喜爱程度"] == "smallDoses"]["雪糕消耗"],
                     c="blue")
    g3 = plt.scatter(x=data[data["喜爱程度"] == "didntLike"]["飞行公里数"], y=data[data["喜爱程度"] == "didntLike"]["雪糕消耗"],
                     c="black")
    plt.legend(handles=[g1, g2, g3], labels=["largeDoses", "smallDoses", "didntLike"], prop={'size': 16})
    plt.savefig("./dataset/dating_pic/飞行时间与雪糕消耗分布图.png")


def draw_raw_data_03(data):
    """
        画出海伦约会数据集中游戏时间比例与雪糕消耗的喜爱成图散点图
        :param data: [1000,4] 矩阵
        :return: None
        """

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    plt.figure(figsize=(20, 9), dpi=100)
    plt.xlabel("游戏时间比例")
    plt.ylabel("雪糕消耗")
    plt.title("海伦约会数据集游戏时间比例与雪糕消耗占比喜爱程度分布图")
    g1 = plt.scatter(x=data[data["喜爱程度"] == "largeDoses"]["游戏时间比例"], y=data[data["喜爱程度"] == "largeDoses"]["雪糕消耗"],
                     c="red")
    g2 = plt.scatter(x=data[data["喜爱程度"] == "smallDoses"]["游戏时间比例"], y=data[data["喜爱程度"] == "smallDoses"]["雪糕消耗"],
                     c="blue")
    g3 = plt.scatter(x=data[data["喜爱程度"] == "didntLike"]["游戏时间比例"], y=data[data["喜爱程度"] == "didntLike"]["雪糕消耗"],
                     c="black")
    plt.legend(handles=[g1, g2, g3], labels=["largeDoses", "smallDoses", "didntLike"], prop={'size': 16})
    plt.savefig("./dataset/dating_pic/游戏时间比例与雪糕消耗分布图.png")


def to_normalize(data):
    """
    给数据做(0,1)归一化
    :param data: [1000,4]
    :return: data_normed
    """
    # minVal_0 = min(data["飞行公里数"])
    # maxVal_0 = max(data["飞行公里数"])
    # print(minVal_0, maxVal_0)
    # minVal_1 = min(data["游戏时间比例"])
    # maxVal_1 = max(data["游戏时间比例"])
    # print(minVal_1, maxVal_1)
    # minVal_2 = min(data["雪糕消耗"])
    # maxVal_2 = max(data["雪糕消耗"])
    # print(minVal_2, maxVal_2)

    # 取得data里的每个特征的最小值、最大值
    minVal = data.min(0)
    maxVal = data.max(0)
    # 各个特征的极差
    ranges = maxVal - minVal
    # print(ranges)
    # 使用numpy生成新矩阵
    new_matrix = numpy.zeros(shape=numpy.shape(data))  # [1000, 3]
    # 做(0,1)归一化
    for i in range(numpy.shape(data)[0]):
        new_matrix[i][0] = (data["飞行公里数"][i] - minVal[0]) / ranges[0]
        new_matrix[i][1] = (data["游戏时间比例"][i] - minVal[1]) / ranges[1]
        new_matrix[i][2] = (data["雪糕消耗"][i] - minVal[2]) / ranges[2]

    return new_matrix


def my_knn(data_normed, sample, labels, k=3):
    """
    实现KNN算法
    :param data_normed: 归一化后的样本集
    :param sample: 需要predict的样本
    :param k: 最近的k个人
    :return: final_label
    """
    # 通过sample数组构建[1000,3]矩阵，然后实现矩阵相减得到new_data_normed
    new_data_normed = tile(sample, (data_normed.shape[0], 1)) - data_normed
    print(tile(sample, (data_normed.shape[0], 1)))
    # 计算欧氏距离
    double_matrix = new_data_normed ** 2
    double_distance = double_matrix.sum(axis=1)
    sqrt_distance = double_distance ** 0.5

    new_matrix = pd.DataFrame()
    new_matrix["distance"] = sqrt_distance
    new_matrix["label"] = labels
    # 排序
    new_matrix = new_matrix.sort_values(by=["distance"], ascending=True)
    # 取前k个
    final_matrix = new_matrix.iloc[:k, :]
    label_dict = {"didntLike": 0, "smallDoses": 0, "largeDoses": 0}
    for i in range(k):
        label_dict[final_matrix.iloc[i]["label"]] += 1

    print(label_dict)
    sorted_label = sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label[0][0]


if __name__ == '__main__':
    file_path = "./dataset/datingTestSet.txt"
    data, target = read_data(file_path)
    # print(data)
    # draw_raw_data_01(data)
    # draw_raw_data_02(data)
    # draw_raw_data_03(data)
    data_normed = to_normalize(data)
    print(data_normed)
    sample = [0.29115949, 0.50910294, 0.51079493]
    label = my_knn(data_normed, sample, target, k=3)
    print("KNN结果是：", label)
