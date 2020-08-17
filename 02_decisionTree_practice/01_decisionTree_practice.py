import math

import numpy as np
import pandas as pd


def load_data():
    """
    加载数据集
    :return: DataFrame
    """
    df = pd.read_csv("./dataSet/01_decisionTree_data.csv")
    return df


def cal_shannonEnt(data_df):
    """
    计算该数据集的香农熵
    :param data_df: 数据集
    :return:
    """
    row_num = len(data_df)
    data_np_array = np.array(data_df)
    label_set = set(data_np_array[:, -1])
    label_set = {label: 0 for label in label_set}
    for row in data_np_array:
        label_set[row[-1]] += 1

    shannonEnt = 0.0
    for key in label_set:
        prob = float(label_set[key]) / row_num
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt


def split_data_set(data_df, index, value):
    """
    删除index和index列下值不等于value的行
    :param data_df: dataFrame
    :param index: 列坐标
    :param value: 比较值
    :return: 剔除后的dataFrame
    """
    # data_np_array = np.array(data_df)
    # res_data_set = []
    # for row in data_np_array:
    #     print(row)
    #     if row[index] == value:
    #         reducedRow = list(row[:index])
    #         reducedRow.extend(row[index + 1:])
    #     res_data_set.append(reducedRow)
    # print(np.array(res_data_set))
    # return np.array(res_data_set)

    # 得到df的列名
    col_name = data_df.columns[index]
    # 取col_name列下等于value的行
    new_df = data_df[data_df[col_name] == value]
    # 删除index列
    new_df = new_df.drop(col_name, axis=1)
    print(new_df)
    return new_df


def majority_cnt(label_list):
    """
    选择出现最多的标签返回
    :param label_list: list
    :return:
    """
    # 用dict计算出不同标签的个数
    label_dict = dict()
    label_set = set(label_list)
    for i in label_set:
        label_dict[i] = 0
    # 遍历list
    for i in label_list:
        label_dict[i] += 1

    # sort这个dict取最大
    sorted_label = sorted(label_dict.items(), key=lambda i: i[0], reverse=False)
    # 返回次数最大的标签
    return sorted_label[0][0]


def build_tree(data_df):
    """
    建树相关的代码
    :param data_df: dataFrame
    :return:
    """
    # 取出标签列表
    label_list = list(data_df["label"])
    # 若这个矩阵的所有标签都一致 则不再分类 直接返回该类标签
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]

    # 若该数据集只剩下一个特征和label列了 那么不再分类
    if len(data_df.iloc[0, :]) == 2:
        return majority_cnt(label_list)

    # 得到最优特征
    best_feature_index = find_best_feature_4_split(data_df)
    # 通过index得到feature_name
    feature_name = data_df.columns[best_feature_index]

    # 初始化树
    tree = {feature_name: {}}
    # 把data_df处理一下 选定了最优特征了就需要把这列特征删除 然后在剩余的矩阵里再递归选择
    unique_val = set(list(data_df[feature_name]))
    data_df.drop(feature_name, axis=1, inplace=True)
    for val in unique_val:
        tree[best_feature_index][val] = build_tree(split_data_set(data_df, best_feature_index, val))
    return tree


def find_best_feature_4_split(data_df):
    """
    找到最合适的特征做分割
    :param data_df: dataFrame
    :return:
    """
    # print(data_df)
    # 求特征数
    feature_num = len(data_df.columns) - 1
    # 计算原矩阵信息熵
    base_shannonEnt = cal_shannonEnt(data_df)
    # print(base_shannonEnt)
    # 最优信息增益值 对应特征index
    best_info_gain, best_feature_index = 0.0, -1
    for index in range(feature_num):
        # 获得该列的list
        col_list = data_df.iloc[:, index]
        col_list = list(col_list)
        # 该列去重
        unique_value = list(set(col_list))
        tmp_ent = 0.0
        for value in unique_value:
            sub_data_set = split_data_set(data_df, index, value)
            # print(sub_data_set)
            prob = len(sub_data_set) / float(len(data_df))
            tmp_ent += prob * cal_shannonEnt(sub_data_set)
        # print(tmp_ent)

        info_gain = base_shannonEnt - tmp_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = index

    """
    经过打印可以看到下面结果：
           nosurfacing  flippers label
    0            1         1   yes
    1            1         1   yes
    2            1         0    no
    3            0         1    no
    4            0         1    no
    # 这是原矩阵的香农熵 可以看到此时熵很大 所以很无序
    0.9709505944546686
    
    # 下面两个小矩阵是针对第一列遍历0,1两种结果取得的子矩阵 split_data_set()函数的作用是除去index列和index列上不等于value的行
       flippers label
    3         1    no
    4         1    no
       flippers label
    0         1   yes
    1         1   yes
    2         0    no
    # 可以看到两个小矩阵的香农熵之和为0.55 小了很多
    0.5509775004326937
    
       nosurfacing label
    2            1    no
       nosurfacing label
    0            1   yes
    1            1   yes
    3            0    no
    4            0    no
    # 同理，这两个矩阵香农熵0.8 比按照第一列的value=0分割混乱（熵大）
    0.8
    
    
    注：信息增益的算法就是原矩阵的香农熵(0.97)-分割后的香农熵(越小则增益越大),增益作为最终选择特征的标准来做判断
    """

    return best_feature_index


if __name__ == '__main__':
    data_df = load_data()
    # cal_shannonEnt(data_df)
    # split_data_set(data_df, 1, 1)
    # find_best_feature_4_split(data_df)
    # build_tree(data_df)
    # majority_cnt(list(data_df["label"]))
    build_tree(data_df)
