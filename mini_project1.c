import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
data={
    'species':['setosa','versicolor','virginica','versicolor','setosa','virginica','setosa'],
    'sepal_length':[4.9,5.7,np.nan,6.3,5.0,6.1,np.nan],
    'sepal_width':[3.5,2.9,2.9,2.7,3.1,3.4,3.3]
}
df=pd.DataFrame(data)
print(df)

print("\nfirst five rows:",df.head())
print("information details",df.info())
print(df.describe())
print("missing values before cleaning")
print(df.isnull().sum())
#filling missing values
df['sepal_length'].fillna(df['sepal_length'].mean(),inplace=True)
print(df)
#data type conversion
df['sepal_length']=df['sepal_length'].astype(int)
print("DataFrame after converting 'sepal_length' to int")
print(df.dtypes)

import numpy as np

sns.histplot(df["sepal_length"],kde=True,bins=20,color="blue")
plt.title("Distribution of sepal length")
plt.show(block=False)

#correlation heatmap
numeric_df=df.select_dtypes(include='number')
corr=numeric_df.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("correlation heatmap")
plt.show(block=False)

# Identify Numerical Variables
# These are usually 'float64' or 'int64'
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# 2. Identify Categorical Variables
# These are usually 'object', 'category', or 'bool'
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical Variables: {numerical_cols}")
print(f"Categorical Variables: {categorical_cols}")

# Quick check of data types
print("\nColumn Data Types:")
print(df.dtypes)
from sklearn.datasets import load_iris

iris = load_iris()
x = range(len(iris.data))

sepal_length = iris.data[:, 0]
sepal_width  = iris.data[:, 1]
petal_length = iris.data[:, 2]
plt.figure()
plt.plot(x, sepal_length, label='Sepal Length')
plt.plot(x, sepal_width, label='Sepal Width')
plt.plot(x, petal_length, label='Petal Length')
plt.xlabel('Samples')
plt.ylabel('Length (cm)')
plt.title('Line Plot of Iris Features')
plt.legend()
plt.show(block=False)
avg_sepal_length = sepal_length.mean()
avg_sepal_width  = sepal_width.mean()
avg_petal_length = petal_length.mean()

features = ['Sepal Length', 'Sepal Width', 'Petal Length']
values = [avg_sepal_length, avg_sepal_width, avg_petal_length]

plt.figure()
plt.bar(features, values)
plt.ylabel('Average Length (cm)')
plt.title('Bar Chart of Iris Features')
plt.show(block=False)
iris = sns.load_dataset("iris")
plt.figure(figsize=(8,5))
sns.boxplot(data=iris[['sepal_length', 'sepal_width', 'petal_length']])
plt.title("Box Plot of Iris Features")
plt.show()
sns.pairplot(
    iris,
    vars=['sepal_length', 'sepal_width', 'petal_length'],
    hue='species'
)
plt.show(block=False)

plt.figure(figsize=(8,5))
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.title("Violin Plot of Sepal Length by Species")
plt.show(block=False)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = sns.load_dataset("iris")

# Features and target
X = iris[['sepal_length', 'sepal_width', 'petal_length']]
y = iris['species']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

#Model Evaluation
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
