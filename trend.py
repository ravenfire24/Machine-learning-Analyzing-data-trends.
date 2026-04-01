import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv("sales_data.csv")

print("\nSales Dataset Preview\n")
print(df.head())

sns.set(style="whitegrid")

# Sales Trend
plt.figure(figsize=(10,6))
sns.lineplot(x="Month", y="Sales", data=df, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales ($)")
plt.show()

# Marketing Spend vs Sales
plt.figure(figsize=(10,6))
sns.scatterplot(x="Marketing Spend", y="Sales", data=df)
plt.title("Marketing Spend vs Sales")
plt.xlabel("Marketing Spend")
plt.ylabel("Sales")
plt.show()

# Holiday Effect
plt.figure(figsize=(8,6))
sns.boxplot(x="Holiday Season", y="Sales", data=df)
plt.title("Holiday Season Impact on Sales")
plt.show()

X = df[["Month", "Marketing Spend", "Holiday Season"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = r2_score(y_test, y_pred) * 100
print(f"Model Accuracy (R² Score): {accuracy:.2f}%")
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_
})

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=coefficients)
plt.title("Feature Impact on Sales")
plt.show()

sample_input = pd.DataFrame({
    "Month": [25],
    "Marketing Spend": [10500],
    "Holiday Season": [1]
})

predicted_sales = model.predict(sample_input)

print(f"Predicted Sales for Month 25: ${predicted_sales[0]:.2f}")
