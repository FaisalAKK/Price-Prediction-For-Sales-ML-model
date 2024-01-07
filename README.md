# Price-Prediction-For-Sales-ML-model
It is a predictive price prediction model based on Melbourne City dataset taken from Kaggle.
<br>
Author -- Muhammad Faisal Anwar Khan
# Property Business Problem
Our business provide house rental and sale services. Customers can rent a place from owners directly from the website. The challenge for our company is to decide the perfect price for a place.
pip install tabulate
<br>
#importing all required libraries
<br>
import pandas as pd
<br>
import numpy as np
<br>
import matplotlib.pyplot as plt
<br>
import seaborn as sns
<br>
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
<br>
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
<br>
from sklearn.linear_model import LinearRegression
<br>
from sklearn.metrics import mean_squared_error, mean_absolute_error
<br>
from sklearn.decomposition import PCA
<br>
from sklearn.neighbors import KNeighborsRegressor
<br>
import warnings
<br>
warnings.filterwarnings("ignore")
<br>
from tabulate import tabulate
<br>

%matplotlib inline

#Importing 1st csv file
Housing1 = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
#Importing 2nd csv file
Housing2 = pd.read_csv('Melbourne_housing_FULL.csv')
# Printing top 5 values of each coloumn
Housing1.head()
# Checking for null values
Housing1.info()
# Calculating mean, max, min and other values
Housing1.describe()
# Calculating median
Housing1.median()
# Plotting Price Using distplot
sns.distplot(Housing1['Price']);
# Plotting Price Using boxplot
sns.boxplot(Housing1['Price']);
# Plotting relation between Price and Regions
plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')
sns.barplot(x = Housing1['Regionname'], y = Housing1['Price'])
plt.title("bar plot for Regionname to Price")
plt.xticks(rotation=45);
# Plotting Relation of Price with Type of property
plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')
sns.boxplot(x = Housing1['Type'], y = Housing1['Price'])
plt.title("bar plot for Type to Price");
# Plotting Relation of Price with area
plt.figure(figsize=(10,7), facecolor='w', edgecolor='b')
sns.scatterplot(x = Housing2["Longtitude"], y = Housing2["Lattitude"], hue= Housing2["Price"], size= Housing2["Landsize"])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)
plt.title("Lattitude to Longitude with Price and Landsize");
# Plotting spread of types of property throughout the area
plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')
sns.scatterplot(x = Housing2["Longtitude"], y = Housing2["Lattitude"], hue= Housing2["Type"])
plt.title("Lattitude to Longitude for Type of room");
# Plotting Relation between some important featuears
plt.figure(figsize=(16, 6))
sns.heatmap(Housing1.corr(), annot=True, cmap='coolwarm');
# Separating different variables and making their columns
Housing1 = pd.concat([Housing1, pd.get_dummies(Housing1["Type"]), pd.get_dummies(Housing1["Method"]), pd.get_dummies(Housing1["Regionname"])], axis=1)

# Removing useless columns
Housing1 = Housing1.drop(["Suburb", "Address", "SellerG", "CouncilArea", "Type", "Method", "Regionname"], 1)

# Converting Date into other format for ease of use
Housing1['Date'] = [pd.Timestamp(x).timestamp() for x in Housing1["Date"]]

Housing1 = Housing1[Housing1['Price'] <= 2500000]
# Droping the null values
Housing1 = Housing1.dropna()

Housing1

# Feature Selection

# Separating Predictors and Responses
X = Housing1.drop("Price", 1)
Y = Housing1["Price"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

# Modeling

# Applying Gradient Boosting technique
gbr = GradientBoostingRegressor(n_estimators=1000, max_depth=5)
gbr.fit(X_train, Y_train)

# Model Evaluation
# Calculating various factors
print("Gradient Boosting Training R^2 Score: ", gbr.score(X_train, Y_train))
print("Gradient Boosting Test R^2 Score: ", gbr.score(X_test, Y_test))
y_pred = gbr.predict(X_test)
print("Mean Squared Error: ", mean_squared_error(y_pred, Y_test))
print("Mean Absolute Error: ", mean_absolute_error(y_pred, Y_test))
