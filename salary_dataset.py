# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:59:49 2024

@author: Joe Amin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
plt.style.use('ggplot')

# Read the CSV file
data_frame = pd.read_csv(r'H:\Position_Salaries.csv')

# Extract features and target variable
x = data_frame['Level'].values.reshape(-1, 1)
y = data_frame['Salary'].values

# Create an instance of StandardScaler and fit_transform the feature
scale = StandardScaler()
x_norm = scale.fit_transform(x)

# Set a wider figure size
plt.figure(figsize=(12, 10))

# Scatter plot for original data
plt.subplot(2, 2, 1)
plt.scatter(x, y, c='g')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Original Data')

# Scatter plot for scaled data
plt.subplot(2, 2, 2)
plt.scatter(x_norm, y, c='g')
plt.xlabel('Level (Scaled)')
plt.ylabel('Salary')
plt.title('Scaled Data')

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x_norm, y)

# Plot the linear regression line
plt.subplot(2, 2, 3)
plt.plot(x_norm, model.predict(x_norm), color='r')
plt.scatter(x_norm, y, c='g')
plt.xlabel('Level (Scaled)')
plt.ylabel('Salary')
plt.title('Linear Regression with Scaled Data')

# Model evaluation for linear regression
print(f'Linear Regression - Mean Squared Error: {mean_squared_error(y, model.predict(x_norm))}')
print(f'Linear Regression - Mean Absolute Error: {mean_absolute_error(y, model.predict(x_norm))}')
print(f'Linear Regression - R2 Score: {r2_score(y, model.predict(x_norm))}')

# Create polynomial features and fit polynomial regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
model2 = LinearRegression()
model2.fit(x_poly, y)
y_poly = model2.predict(x_poly)

# Plot the polynomial regression curve
plt.subplot(2, 2, 4)
plt.plot(x, y_poly, color='r')
plt.scatter(x, y, c='g')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.title('Polynomial Regression')

# Model evaluation for polynomial regression
print(f'Polynomial Regression - Mean Squared Error: {mean_squared_error(y, model2.predict(x_poly))}')
print(f'Polynomial Regression - Mean Absolute Error: {mean_absolute_error(y, model2.predict(x_poly))}')
print(f'Polynomial Regression - R2 Score: {r2_score(y, model2.predict(x_poly))}')


plt.show()
