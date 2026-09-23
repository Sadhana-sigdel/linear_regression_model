from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import math

df = pd.read_csv("data/house_prices.csv")
print(df.head())
print(df.shape)
print(df.dtypes)

X = df.drop("price", axis=1)
y = df["price"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual prices:", y_test.values)


loss = mean_squared_error(y_test, y_pred)
print("MSE:", loss)

rmse = math.sqrt(loss)
print("RMSE:", rmse)

r2 = r2_score(y_test, y_pred)
print("R2:", r2)