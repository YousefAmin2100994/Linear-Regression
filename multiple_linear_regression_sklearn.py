# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:28:48 2024

@author: Joe Amin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV file
data_frame = pd.read_csv(r'H:\FuelConsumptionCo2.csv')

# Extract features (independent variables) and target variable
x = np.asanyarray(data_frame[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(data_frame[['CO2EMISSIONS']])

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(x_train, y_train)

# Print the coefficients and intercept
print("Coefficients:", model.coef_[0])
print("Intercept:", model.intercept_[0])

# Make predictions on the test set
predictions = model.predict(x_test)

# Evaluate the model's performance on the test set
MAE = mean_absolute_error(y_test, predictions)
MSE = mean_squared_error(y_test, predictions)
print("Mean Absolute Error on Test Set:", MAE)
print("Mean Squared Error on Test Set:", MSE)
