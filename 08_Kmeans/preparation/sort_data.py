import pandas as pd


def txt_2_csv(file_path):
    df = pd.read_table(file_path, header=None, sep='\t')
    # 控制小数位数
    df.iloc[:, :] = df.iloc[:, :].round(6)
    df.to_csv("../dataSet/data2.csv", sep=',', header=["x1", "x2"], index=False)


if __name__ == '__main__':
    # 把txt数据转为csv
    file_path = "../dataSet/testSet2.txt"
    txt_2_csv(file_path)
