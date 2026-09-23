from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("data/house_prices.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
