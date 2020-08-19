from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get data from this initial folder
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# defining initial workbook with csv file
initial_workbook = os.path.join(THIS_FOLDER, "Salary_Data.csv")

# importing data into pandas
dataset = pd.read_csv(initial_workbook)

# creating dataset for matrix of features
X = dataset.iloc[:, :-1].values

# creating dataset for variable vector
y = dataset.iloc[:, -1].values

# spliting data from
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# using linear regression model and building it on train data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting with regressor on test data
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experiance (Training set)')
plt.xlabel('Years of experiance')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experiance (Test set)')
plt.xlabel('Years of experiance')
plt.ylabel('Salary')
plt.show()