import pandas as pd
import numpy as np

#先对数据进行转置，之后再在r语言的模型中进行补值
df0 = pd.read_csv("data/52529.csv", index_col=0)
df1 = pd.read_csv("data/74672.csv", index_col=0)
df2 = pd.read_csv("data/75748.csv", index_col=0)

# print(df0.shape)
# print(df1.shape)
# print(df2.shape)

df0 = df0.T
df1 = df1.T
df2 = df2.T

# print(df0.shape)
# print(df1.shape)
# print(df2.shape)

df0.to_csv("data-sci/52529.csv")
df1.to_csv("data-sci/74672.csv")
df2.to_csv("data-sci/75748.csv")



