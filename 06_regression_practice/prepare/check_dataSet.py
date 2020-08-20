import pandas as pd

file_path = "../dataSet/3_1.csv"
df = pd.read_csv(file_path)
print(df)
print(df.describe())