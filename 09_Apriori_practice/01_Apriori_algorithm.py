import itertools


def load_data(file_path):
    """
    加载数据集
    :param file_path: 文件路径
    :return:  data_list list类型
    """

    data_list = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            data_list.append(line.split(','))
    # print(data_list)
    return data_list


def data_2_index(data_set):
    """
    把data_set中的字符串转为index
    :param data_set: 数据列表 list
    :return: data_set 数据列表 list
    """
    # 把data_set拆包 然后取出其中的元素 通过set去重
    items = set(itertools.chain(*data_set))
    # print(items)
    # 保存字符串到编号的映射
    str_2_index = {}
    # 保存编号到字符串的映射
    index_2_str = {}
    for index, item in enumerate(items):
        # print(index, '->', item)
        str_2_index[item] = index
        index_2_str[index] = item

    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            data_set[i][j] = str_2_index[data_set[i][j]]

    # print(data_set)
    return data_set


def build_c1(data_set):
    """
    创建候选1项集
    :param data_set: 数字化后的data_set
    :return:
    """
    # 把data_set中的元素去重
    items = set(itertools.chain(*data_set))
    # print(items)
    # 用frozenset把项集装进新列表里
    """
    Tips： 使用frozenset的原意是接下来的步骤需要使用items里的内容做key
    若直接将数字作为key的话也可以，但是后面还有生成二项集、三项集的操作，那就需要用list等来装，这样就不能作为key了
    
    即：
        my_dict = {}
        my_dict[frozenset([1, 2, 3])] = 2.2
        这个操作时可以的，打印my_dict是：{frozenset({1, 2, 3}): 2.2}
        
        my_dict = {}
        my_dict[[1, 2, 3]] = 2.2
        这个非操作是非法的，TypeError: unhashable type: 'list' 即list不能哈希
        
    
    当然，办法总比困难多，我试过将list转为str，将字符串作为key放入dict。这样也是可以，但是需要两个函数专门处理，
    并且这两个解析函数还需要根据不同的数据类型专门写。
    """
    frozen_items = [frozenset(i) for i in enumerate(items)]
    # print(frozen_items)
    return frozen_items


def ck_2_lk(data_set, ck, min_support):
    """
    根据候选k项集生成频繁k项集，依据min_support
    :param data_set: 数据集 list类型
    :param ck: 候选k项集 list类型，list装frozenset
    :param min_support: float 最小支持度
    :return: lk dict类型
    """

    # 频数字典 用来记录每个项集出现的频数
    support = {}
    # 用数据集的每一行跟候选项集的每个项对比，若该项集是其中子集，则+1，否则为0
    for row in data_set:
        for item in ck:
            if item.issubset(row):
                support[item] = support.get(item, 0) + 1
    # print(support)
    # 计算频率需要用到长度
    length = len(data_set)
    lk = {}
    for key, value in support.items():
        # print(key, value)
        percent = value / length
        # 频率大于最小支持度才能进入频繁项集
        if percent > min_support:
            lk[key] = percent

    return lk


def lk_2_ck_plus_1(lk):
    """
    将频繁k项集（lk）转为候选k+1项集
    :param lk: 频繁k项集 dict
    :return: ck_plus_1
    """
    lk_list = list(lk)
    # 保存组合后的k+1项集
    ck_plus_1 = set()
    lk_size = len(lk)
    # 若lk_size<=1则不需要再组合
    if lk_size > 1:
        # 获取频繁项集的长度
        k = len(lk_list[0])
        """
        itertools.combinations(range(lk_size), 2) 相当于从lk_size中任选2个项集 i,j
        即c_n_2
        """
        for i, j in itertools.combinations(range(lk_size), 2):
            # print(i, j)
            t = lk_list[i] | lk_list[j]
            # 两两组合后项集长度是k+1，否则不要
            if len(t) == k + 1:
                ck_plus_1.add(t)
    # print(ck_plus_1)
    return ck_plus_1


def get_all_L(data_set, min_support):
    """
    把所有的频繁项集拿到
    :param data_set: 数据
    :param min_support:  最小支持度
    :return:
    """
    # 创建候选1项集
    c1 = build_c1(data_set)
    # 从候选1项集 到 频繁1项集
    l1 = ck_2_lk(data_set, ck=c1, min_support=0.05)
    L = l1
    Lk = l1
    while len(Lk) > 0:
        lk_key_list = list(Lk.keys())
        # 频繁k 到 候选k+1
        ck_plus_1 = lk_2_ck_plus_1(lk_key_list)
        # 候选k 到 频繁k
        Lk = ck_2_lk(data_set, ck_plus_1, min_support)
        if len(Lk) > 0:
            L.update(Lk)
        else:
            break
    return L


def rules_from_item(item):
    # 关联规则左边
    left = []
    for i in range(1, len(item)):
        """
        若使用append 则会把combinations对象原封不动添加进列表里
        使用extend 则会把combinations对象拆包再添加到列表里
        combination对象是可以迭代的一个对象 combinations(item,1) = combinations((1,),(2,),(3,))
        """
        left.extend(itertools.combinations(item, i))
        # left.append(itertools.combinations(item, i))

    return [(frozenset(i), frozenset(item.difference(i))) for i in left]


def rules_from_L(L, min_confidence):
    # 保存所有候选的关联规则
    rules = []
    for Lk in L:
        # 频繁项集长度要大于1才能生成关联规则
        if len(Lk) > 1:
            rules.extend(rules_from_item(Lk))
    result = []
    for left, right in rules:
        # left和right都是frozenset类型 二者可以取并集 然后L里去查询支持度
        support = L[left | right]
        # 置信度公式
        confidence = support / L[left]
        lift = confidence / L[right]
        if confidence > min_confidence:
            result.append({"左": left, "右": right, "支持度": support, "置信度": confidence, "提升度": lift})

    return result


if __name__ == '__main__':
    file_path = "./dataSet/data.txt"
    # 加载数据
    data_set = load_data(file_path)
    # 把数据转为数字 方便比较计算
    data_set = data_2_index(data_set)
    # # 创建候选1项集
    # c1 = build_c1(data_set)
    # # 从候选1项集 到 频繁1项集
    # l1 = ck_2_lk(data_set, ck=c1, min_support=0.05)
    # # 从频繁1项集 到 候选2项集
    # c2 = lk_2_ck_plus_1(l1)
    # # 从候选2项集 到 频繁2项集
    # l2 = ck_2_lk(data_set, c2, 0.05)
    # print(l2)
    # # 从频繁2项集 到 候选3项集
    # c3 = lk_2_ck_plus_1(l2)
    # print(c3)
    # l3 = ck_2_lk(data_set, c3, 0.05)
    # print(l3)
    # 得到所有频繁项集
    L = get_all_L(data_set, 0.05)
    # 得到所有关联规则
    result = rules_from_L(L, min_confidence=0.05)
    print(result)
