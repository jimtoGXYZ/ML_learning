import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    df = pd.read_csv(file_path)

    return df


def cal_distance(point_A, point_B):
    point_A = point_A.A
    point_B = point_B.A
    distance = np.sqrt(np.sum(np.power((point_A - point_B), 2)))

    return distance


def my_paint(df):
    plt.figure(figsize=(10, 10), dpi=180)
    plt.scatter(x=df['x1'], y=df['x2'])
    plt.savefig("./dataSet/pic/origin_data.png")


def create_centers(data_matrix, k=4):
    # 先把数计算出来降低时间复杂度
    n, m = np.shape(data_matrix)
    # 根据数据集创建随机质心 shape=[k,m] 保存质心的坐标
    center_matrix = np.zeros(shape=(k, m))
    for i in range(m):
        min_val, max_val = min(data_matrix[:, i])[0, 0], max(data_matrix[:, i])[0, 0]
        center_matrix[:, i] = np.array((min_val + (max_val - min_val) * np.random.rand(k, 1)))[:, 0]
    center_matrix = np.mat(center_matrix)
    return center_matrix


def my_KMeans(data_matrix, k=4):
    # 创建分类矩阵[n,2] n个样本 2列用于存质心index和到其对应质心的distance
    cluster_matrix = np.zeros(shape=(len(data_matrix), 2))
    # 创建随机质心矩阵
    center_matrix = create_centers(data_matrix, k)
    # 得到两个矩阵的shape
    center_matrix_n, center_matrix_m = np.shape(center_matrix)
    data_matrix_n, data_matrix_m = np.shape(data_matrix)

    # 是否有变化flage
    change_flag = True
    # 直到没有变化为止结束
    while change_flag:
        change_flag = False
        # 计算每一个点到每一个质心距离 取最小值归类
        for i in range(data_matrix_n):
            min_dist, min_index = math.inf, -1
            for j in range(center_matrix_n):
                # 计算距离
                new_distance = cal_distance(data_matrix[i], center_matrix[j])
                # 更新最近距离和质心index
                if min_dist > new_distance:
                    min_dist = new_distance
                    min_index = j
            # 若质心有变化 更新质心index
            if min_index != cluster_matrix[i, 0]:
                print("min_index != cluster_matrix[i, 0]", cluster_matrix[i, 0], min_index)
                cluster_matrix[i, 0] = min_index
                cluster_matrix[i, 1] = new_distance
                change_flag = True
        # 根据分类更新质心位置
        for center in range(k):
            # 从data_matrix中找到该类点 返回矩阵
            # print(data_matrix[cluster_matrix[:, 0] == center])
            # 对该矩阵求平均 axis表方向 0为列求平均 最后[1,m]
            # print("avg:", np.average(data_matrix[cluster_matrix[:, 0] == center], axis=0))
            center_matrix[center] = np.average(data_matrix[cluster_matrix[:, 0] == center], axis=0).A[0]

    return center_matrix


def my_paint_with_center(center_matrix, df):
    plt.figure(figsize=(10, 10), dpi=180)
    plt.scatter(x=df['x1'], y=df['x2'], c='blue')
    center_df = pd.DataFrame(center_matrix)
    print(center_df)
    plt.scatter(x=center_df.iloc[:, 0], y=center_df.iloc[:, 1], c='red', marker='x')
    plt.savefig("./dataSet/pic/after_kmeans_data.png")
    plt.show()


if __name__ == '__main__':
    file_path = "./dataSet/data1.csv"
    # 读数据
    df = load_data(file_path)
    # 画图
    # my_paint(df)
    # 转df为矩阵
    data_matrix = np.mat(df)
    # Kmeans算法分类
    center_matrix = my_KMeans(data_matrix, 4)
    # 画图2
    my_paint_with_center(center_matrix, df)
