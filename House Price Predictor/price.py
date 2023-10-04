import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
from scipy import stats
from scipy.stats import norm , skew

from sklearn.datasets import fetch_california_housing
fetch_california_housing = fetch_california_housing()
x = fetch_california_housing.data
y = fetch_california_housing.target

data = pd.DataFrame(x, columns = fetch_california_housing.feature_names)
data["SalePrice"] = y
print(data.head())

describe = fetch_california_housing.DESCR
print(data.shape)
print(data.info())
print(data.describe())

sns.distplot(data["SalePrice"])
plt.show()

skw = data["SalePrice"].skew()
kurt = data["SalePrice"].kurtosis()
print(f"Skewness = {skw}")
print(f"Kurtosis = {kurt}")

sns.distplot(data["SalePrice"] , fit = norm)

mu , sigma = norm.fit(data["SalePrice"])
print(f"mu = {mu} and sigma = {sigma}")

plt.legend([f"Normal Distribution: mu = {mu} and sigma = {sigma}"], loc = "best")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.show()

#QQ plot
fig = plt.figure()
res = stats.probplot(data["SalePrice"],plot = plt)
plt.show()

#To remove outliers
data["SalePrice"] = np.log1p(data["SalePrice"])

# plotting again
sns.distplot(data["SalePrice"] , fit = norm)

mu , sigma = norm.fit(data["SalePrice"])
print(f"mu = {mu} and sigma = {sigma}")

plt.legend([f"Normal Distribution: mu = {mu} and sigma = {sigma}"], loc = "best")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.show()

#QQ plot
fig = plt.figure()
res = stats.probplot(data["SalePrice"],plot = plt)
plt.show()

#Correlation
plt.figure(figsize=(10,10))
corr = data.corr()
sns.heatmap(corr , annot= True , cmap = plt.cm.PuBu)
plt.show()

cor_target = abs(corr["SalePrice"]) #absolute value
relevant_features = cor_target #highly correlated
names = [] # getting names of features
for index in relevant_features:
    names.append(index)
print(len(names))

#Model Building
from sklearn.model_selection import train_test_split
x = data.drop("SalePrice", axis= 1)
y = data["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train , y_train)

predictions = lr.predict(x_test)

print(f"Actual Value of house = {y_test[0]}")
print(f"Model Predicted value = {predictions[0]}")

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test , predictions)
rmse = np.sqrt(mse)
print(f"Accuracy: {rmse}")
