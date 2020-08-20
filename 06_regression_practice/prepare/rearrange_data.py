import pandas as pd

df = pd.read_csv("../dataSet/3.csv", sep="\t")
print(df)
df.to_csv("../dataSet/3_1.csv", sep=",", index=False, header=["x0", "x1", "y"])
