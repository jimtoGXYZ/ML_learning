import os
import operator
import pandas as pd
import numpy as np


def read_data(file_path):
    """
    读取指定目录下的所有txt文件
    并且[32,32] -> [1024,1] 的data转换
    并且获取每一个文件的target 做成一个向量
    :param file_path: 需要读取的文件夹路径
    :return: DataFrame[data,target]
    """
    dir_list = os.listdir(file_path)
    data = pd.DataFrame(columns=("data", "target"))
    for file_name in dir_list:
        # 获得目标值
        file_target = file_name.split(sep="_")[0]
        # 读取txt文件，转为[1024,1]
        data_32 = pd.read_table(filepath_or_buffer=file_path + file_name, header=None)
        # df格式转为np.array
        data_32 = np.array(data_32)
        new_str = ""
        for i in data_32:
            new_str += i[0]
        # 向dataframe中加一行
        data = data.append({"data": new_str, "target": file_target}, ignore_index=True)

    return data


def df_to_np(data):
    """
    将df对象矩阵转为np.array矩阵
    :param data: df类型 ["data","target"]
    :return: np.array
    """
    data = data["data"]
    # 需要先做一个0向量才能生成
    np_matrix = np.array(np.zeros(shape=(1, 1024)))
    # 先构造一个np矩阵 因为最小0 最大1 所以不需要归一化了
    for item in data:
        item_list = []
        for index in item:
            item_list.append(int(index))
        item_nparray = np.array(item_list)
        np_matrix = np.row_stack((np_matrix, item_nparray))
    np_matrix = np_matrix[1:]
    # 把第一行全0除去
    return np_matrix


def my_knn(line, train_matrix, train_label, k):
    np_matrix = np.tile(line, (train_matrix.shape[0], 1)) - train_matrix
    # 计算欧式距离
    double_matrix = np_matrix ** 2
    double_distance = double_matrix.sum(axis=1)
    distance = double_distance ** 0.5
    new_matrix = pd.DataFrame()
    new_matrix["distance"] = distance
    new_matrix["label"] = train_label
    # 排序
    new_matrix = new_matrix.sort_values(by=["distance"], ascending=True)
    # 取前k个
    final_matrix = new_matrix.iloc[:k, :]
    label_dict = {}
    for i in range(10):
        label_dict[str(i)] = 0
    for i in range(k):
        label_dict[final_matrix.iloc[i]["label"]] += 1
    sorted_label = sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_label[0][0])
    return sorted_label[0][0]


def cal_acc_recall(train_data, test_data, k=3):
    """
    通过train_data来实现knn，然后遍历test_data来算acc recall
    :param train_data: 训练集
    :param test_data: 测试集
    :param k: k个最临近值
    :return: None
    """
    train_matrix = df_to_np(train_data)
    train_label = train_data["target"]
    test_matrix = df_to_np(test_data)
    test_label = test_data["target"]
    print(test_label[0])
    # 计数器
    iter = 0
    # acc_num
    acc_num = 0
    for line in test_matrix:
        predict_label = my_knn(line, train_matrix, train_label, k)
        acc_num = acc_num + 1 if predict_label == test_label[iter] else acc_num
        iter += 1
    # 准确率
    print("准确率是%f" % (acc_num / (iter + 1)))
    # 召回率 懒得做了
    return None


if __name__ == '__main__':
    # 1' 读取数据集、返回data，target
    trainingDigits_path = "./dataset/knn-handwriting_num/trainingDigits/"
    testDigits_path = "./dataset/knn-handwriting_num/testDigits/"
    train_data = read_data(trainingDigits_path)
    # 2' 读取测试数据集
    test_data = read_data(testDigits_path)
    # 3' 使用KNN对test_data一一测试
    cal_acc_recall(train_data, test_data, 3)
