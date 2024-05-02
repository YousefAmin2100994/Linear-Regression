# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 17:04:50 2024

@author: Joe Amin
"""

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.linear_model import LinearRegression

# Read data
data_frame = pd.read_csv(r'H:\train.csv')
data_frame.drop(213, axis=0, inplace=True)

# Extract input (x) and output (y) variables
x = data_frame['x'].values.reshape(-1, 1)  # Reshape to a 2D array
y = data_frame['y'].values

# Create and fit the model
model = LinearRegression()
model.fit(x, y)
y_pred=model.predict(x)

# Scatter plot and regression line
plt.figure(figsize=(12, 8))

# Scatter plot
plt.subplot(2, 2, 1)
plt.scatter(x, y, c='green', label='Points')
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Regression line
plt.subplot(2, 2, 2)
plt.scatter(x, y, c='green', label='Points')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Predicted Y')
plt.legend()

# Histogram
plt.subplot(2, 2, 3)
plt.hist(y_pred-y, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Residuals Values')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Scatter
plt.subplot(2, 2, 4)
plt.scatter(y,y_pred, c='green')
plt.title('Scatter Plot')
plt.xlabel('Y')
plt.ylabel('Y_pred')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

#testing
print(model.predict([[19.5]])[0])
print(model.coef_[0])
print(model.intercept_)
