import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1. DATA LOADING
# =========================
df = pd.read_csv("Salary_Data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# =========================
# 2. DATA CLEANING
# =========================
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# =========================
# 3. EXPLORATORY DATA ANALYSIS
# =========================
print("\nStatistical Summary:")
print(df.describe())

# =========================
# 4. DATA VISUALIZATION
# =========================

# Histogram
plt.figure()
plt.hist(df["Salary"], bins=20)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Count")
plt.show()

# Scatter Plot
plt.figure()
plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()

# Box Plot
plt.figure()
sns.boxplot(y=df["Salary"])
plt.title("Box Plot of Salary")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =========================
# 5. PREDICTIVE MODELING
# =========================
X = df[["YearsExperience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 6. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. MODEL EVALUATION
# =========================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# =========================
# 8. REGRESSION LINE VISUALIZATION
# =========================
plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Experience vs Salary")
plt.show()
