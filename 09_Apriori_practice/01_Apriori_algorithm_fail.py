import time


def load_data():
    data_array = [[1, 3, 4],
                  [2, 3, 5],
                  [1, 2, 3, 5],
                  [2, 5]]
    return data_array


def create_C1(data_array):
    """
    创建C1集合
    """
    c1 = []
    for transaction in data_array:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])

    # 将数组递增排序
    c1.sort()
    # 转为frozenset格式的list 为后面frozenset_list作为字典索引做准备 否则很多数据结构是无法作为set的索引的
    # c1 = map(frozenset, c1)
    # tips: 22行操作发现问题： 这个map遍历后指针是回不去的 即二轮嵌套循环第一次把map遍历完后第二轮指针是回不到头部的
    # print(c1)
    return c1


# def scanD(D, ck, min_support):
#     """
#     计算候选集的支持度 返回支持度大于min_support的数据
#     """
#     # 存放数据集ck的频率 如 a:10 b:5
#     cnt = {}
#     for line in D:
#         for can in ck:
#             print(can)
#             print(line)
#             # subset的意思是子集 即can的每个元素都在line中
#             if can.issubset(line):
#                 print(type(can))
#                 print(type(line))
#                 if can not in cnt.keys():
#                     cnt[can] = 1
#                 else:
#                     cnt[can] += 1
#
#     num_of_lines = float(len(D))
#     rest_list = []
#     support_data = {}
#     for key in cnt:
#         # 支持度 = 候选项（key）的出现次数/ 所有数据集的数量
#         support = cnt[key] / num_of_lines
#         if support > min_support:
#             # 头插
#             rest_list.insert(0, key)
#
#         # 存储key对应的支持度
#         support_data[key] = support
#     return rest_list, support_data


def list_2_str(list):
    return str(list)


def str_2_list_4_nums(str):
    str = str.replace('[', '').replace(']', '')
    num_list = str.split(',')
    ans = [int(i) for i in num_list]
    # print(ans)
    return ans


def cal_support(data_array, candidate_array, min_support):
    cnt_set = {}
    for data_line in data_array:
        for candidate in candidate_array:
            #  因为都是经过去重的list 所以可以转成set来作比较  A<B 表示A的元素全都包含在B中 A<=B表示A要小于等于B集
            if set(candidate) <= set(data_line):
                # 把list转为str作为键放入cnt_set
                candidate_str = list_2_str(candidate)
                if candidate_str not in set(cnt_set.keys()):
                    cnt_set[candidate_str] = 1
                else:
                    candidate_str = list_2_str(candidate)
                    cnt_set[candidate_str] += 1

    # 装合格的list
    qualified_list = []
    data_array_length = float(len(data_array))
    # 装每个list对应的支持度
    support_dict = dict()
    for key in cnt_set:
        # print(key,type(key))
        support_val = cnt_set[key] / data_array_length
        support_dict[key] = cnt_set[key] / data_array_length
        if support_val >= min_support:
            qualified_list.insert(0, str_2_list_4_nums(key))

    # print(support_dict)
    # print(qualified_list)
    # print(cnt_set)
    return qualified_list, support_dict


def fix_2(last_list, k):
    """
    把last_list里的list两两组合

    还有问题 这里没有去重
    """
    list_length = len(last_list)
    new_list = []
    for i in range(list_length):
        for j in range(i + 1, list_length):
            tmp_list = list(set(last_list[i] + last_list[j]))
            # 对new_list还要做一次去重 因为可能拼接后的list会重复
            if tmp_list not in new_list:
                new_list.append(tmp_list)
    # print(new_list)
    return new_list


def apriori_cal_support():
    data_array = load_data()
    c1_set = create_C1(data_array)
    # 第一轮把合格的长度1的list筛选出来
    qualified_list, support_dict = cal_support(data_array, c1_set, min_support=0.5)
    total_dict = support_dict
    total_qualified_list = qualified_list
    k = 2
    # 当大于指定最小支持度的list存在的时候就执行
    while len(qualified_list) > 0:
        # 两两组合
        fixed_array = fix_2(qualified_list, k=k)
        # 计算支持度和qualified_list
        qualified_list, support_dict = cal_support(data_array, fixed_array, min_support=0.5)
        # 把qualified_list收集起来 放入公共的list
        total_dict.update(support_dict)
        total_qualified_list += qualified_list
        # print("total:", total_dict)
        k += 1
        # time.sleep(0.5)

    print("total qualified list:", total_qualified_list)
    print("total support dict:", total_dict)
    return total_qualified_list, total_dict


#
# def cal_confident(item_list, total_support_dict, min_confident):
#     """
#     计算可信度
#     可信度的公式是：
#     confident(a->b) = support(a|b)/support(a)
#
#     item_list : 需要计算可信度的项集，长度>=2
#     total_support_dict : 所有组合的支持度字典，key为str(list)
#     """
#     tmp_confident_dict = dict()
#     for item in item_list:
#         confident = total_support_dict[str(item_list)] / total_support_dict[str([item])]
#         # 判断可信度阈值
#         if confident > min_confident:
#             tmp_confident_dict[str([[i for i in item_list if i != item], '->', [item]])] = confident
#
#     # print(tmp_confident_dict)
#
#     return tmp_confident_dict


def cal_confident(qualified_list, pre_list, total_support_dict):
    """
        计算可信度
        可信度的公式是：
        confident(a->b) = support(a|b)/support(a)
    """
    # print("cal_confident:", qualified_list, pre_list)
    confident = total_support_dict[str(qualified_list)] / total_support_dict[str(pre_list)]
    return confident


#
# def build_rules(qualified_list, total_support_dict, pre_list, my_relation_dict, min_confident):
#     """
#
#     qualified_list : 频繁项集的元素去重的list
#     total_support_dict : 全部元素组合的支持度dict
#     pre_list : 上次分配的结果list
#     min_confident : 要求的可信度
#     """
#     # 存放关系的字典 格式 "[[a],'->',[b]] : confident"
#     relation_dict = my_relation_dict
#
#     # set元素一定要大于等于2 否则无法分配
#     for item in qualified_list:
#         # 当左边数组长度为1的时候 就不再执行了 因为再取就是全集
#         if len(qualified_list) == 1:
#             return relation_dict
#
#         # 顺序选择一个item放入箭头右方
#         pre_list.append(item)
#         if len(pre_list) != 0:
#             pre_list = list(set(pre_list))
#
#         # 将qualified_list的元素去除item
#         qualified_list = [i for i in qualified_list if i != item]
#         # 计算可信度 qualified_list -> pre_list  表示箭头左能导致右  total_support_dict用于查询支持度
#         confident = cal_confident(qualified_list, pre_list, total_support_dict)
#         # 判断可信度是否达标
#         if confident > min_confident:
#             # 达标的放入字典里 格式是 "[[a],'->',[b]] : confident"
#             relation_dict[str([qualified_list, '->', pre_list])] = confident
#             print("add:", relation_dict)
#             # 递归
#             ans = build_rules(qualified_list, total_support_dict, pre_list, relation_dict, min_confident)
#             if ans is not None:
#                 relation_dict.update(ans)
#         else:
#             print("可信度不达标：", confident)
#             # 若不达标 则其子集都不会达标的 所以直接退出while
#             return
#     return relation_dict

def build_rules(qualified_list, total_support_dict, right_list, min_confident):
    for item in qualified_list:
        if len(qualified_list) == 1:
            return
        qualified_list = [i for i in qualified_list if i != item]
        if item in right_list:
            continue
        right_list.append(item)
        print(qualified_list, right_list)
        build_rules(qualified_list, total_support_dict, right_list, min_confident)


def find_rules(total_qualified_list, total_support_dict, min_confident):
    """
    找关联规则
    因为频繁项集才能产生关联规则 所以只需要频繁项集 不需要所有项集
    """
    # 1.1 计算频繁项集总元素个数
    qualified_set = set()
    for i in total_qualified_list:
        qualified_set.update(set(i))

    qualified_list = list(qualified_set)
    right_list = []
    relation_dict = build_rules(qualified_list, total_support_dict, right_list, min_confident)
    return relation_dict


if __name__ == '__main__':
    # 找频繁项集
    total_qualified_list, total_support_dict = apriori_cal_support()
    # 找关联规则
    relation_dict = find_rules(total_qualified_list, total_support_dict, 0.6)
    print("x" * 20, relation_dict)
