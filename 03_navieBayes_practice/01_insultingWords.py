import numpy as np


def loadDataSets():
    """
    加载数据集
    :return: dataMatrix,labelList
    """
    dataMatrix = [
        ["stop", "fuck", "you", "bitch", "garbage"],
        ["useless", "dog", "stupid", "worthless"],
        ["suck", "my", "dick", "bitch", "pig", "asshole"],
        ["son", "bitch", "hoocker", "happy"],
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
    ]
    labelList = [1, 1, 1, 1, 0, 0, 0]

    return dataMatrix, labelList


def buildWordsSet(dataMatrix):
    """
    创建单词集合
    :param dataMatrix: 单词矩阵
    :return:
    """
    wordsSet = set([])
    for comment in dataMatrix:
        wordsSet = wordsSet | set(comment)
    return list(wordsSet)


def getOneZeroVector(wordSet, comment):
    """
    以wordSet这个训练集词频向量为依据构建comment的词频（0,1）向量
    :param wordSet:
    :param comment:
    :return:
    """
    one_zero_vec = [0] * len(wordSet)
    for word in comment:
        if word in wordSet:
            one_zero_vec[wordSet.index(word)] = 1
        else:
            print(" %s 没有收录" % word)
    return one_zero_vec


def getBayesParams(one_zero_matrix, labelList):
    """
    计算贝叶斯公式参数
    :param one_zero_matrix: （0,1）词频矩阵
    :param labelList: 每个评论的标签
    :return: log（词频/好类总词数） ， log（词频/坏类总次数） ， 侮辱性评论占训练集总评论概率
    """
    # 侮辱性评论概率
    p_c1 = sum(labelList) / float(len(labelList))
    # 训练集总词数
    total_words_count = len(one_zero_matrix[0])
    # 两种单词出现频率列表
    p0List = np.ones(len(one_zero_matrix[0]))
    p1List = np.ones(len(one_zero_matrix[0]))
    # 计算两类词频
    p0num = 1.0
    p1num = 1.0

    # 遍历所有测试集评论
    for i in range(len(labelList)):
        # 若该评论是侮辱性
        if labelList[i] == 1:
            p1List += one_zero_matrix[i]
            p1num += sum(one_zero_matrix[i])
        else:
            p0List += one_zero_matrix[i]
            p0num += sum(one_zero_matrix[i])
    # 每个词词频列表/该类别词频 再取对数
    p1vec = np.log(p1List / p1num)  # 已知是侮辱性评论情况下，每个词出现的概率
    p0vec = np.log(p0List / p0num)  # 已知不是侮辱性评论情况下，每个词出现的概率

    return p1vec, p0vec, p_c1


def classifyByBayes(p1vec, p0vec, p_c1, one_zero_vector):
    """
    使用贝叶斯参数比较得出结果
    :param p1vec:
    :param p0vec:
    :param p_c1:
    :param one_zero_vector:
    :return:
    """
    # sum(one_zero_vector * p1vec) 对应元素相乘相加
    # p_1 = sum(one_zero_vector * p1vec) + np.log(p_c1)
    p_1 = sum(one_zero_vector * p1vec)
    # p_0 = sum(one_zero_vector * p0vec) + np.log(1.0 - p_c1)
    p_0 = sum(one_zero_vector * p0vec)
    if p_1 > p_0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # 1' 加载数据集
    dataMatrix, labelList = loadDataSets()
    # print(dataMatrix)
    # print(labelList)
    # 2' 创建单词集合
    wordSet = buildWordsSet(dataMatrix)
    # print(wordSet)
    # 3' 创建单词（0,1）矩阵
    one_zero_matrix = []
    for comment in dataMatrix:
        one_zero_vec = getOneZeroVector(wordSet, comment)
        one_zero_matrix.append(one_zero_vec)
    # print(np.array(one_zero_matrix))
    # 4' 得到bayes相关数据
    p1vec, p0vec, p_c1 = getBayesParams(np.array(one_zero_matrix), np.array(labelList))
    # 5'自定义数据测试
    testVec = ["love", "my", "dog", "dalmation"]
    docVec = np.array(getOneZeroVector(wordSet, testVec))
    ans = classifyByBayes(p1vec, p0vec, p_c1, docVec)
    print(ans)
    testVec = ["stupid", "garbage"]
    docVec = np.array(getOneZeroVector(wordSet, testVec))
    ans = classifyByBayes(p1vec, p0vec, p_c1, docVec)
    print(ans)
