import pandas as pd

df = pd.read_csv("./dataset/full_dataset.csv")
df = df.iloc[:, 1:]
df = df.iloc[::2]
df.to_csv("./dataset/final_dataset.csv", index=False)
