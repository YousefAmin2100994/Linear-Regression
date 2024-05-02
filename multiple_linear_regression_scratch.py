# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:03:43 2024

@author: Joe Amin
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        total_cost += (np.dot(w, x[i]) + b - y[i]) ** 2
    return total_cost / (2 * m)

def compute_gradient(x, y, w, b):
    dj_dw = np.zeros(len(w))
    dj_db = 0
    m = x.shape[0]
    for i in range(m):
        dj_dw += (np.dot(w, x[i]) + b - y[i]) * x[i]
        dj_db += (np.dot(w, x[i]) + b - y[i])
    return dj_dw / m, dj_db / m

def gradient_decent(x, y, w, b, alpha=5.0e-7, number_of_iterations=1000):
    for i in range(number_of_iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b -= alpha * dj_db
    return w, b

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 0
w_init = np.array([0,0,0,0])




# Apply gradient descent
result_w, result_b = gradient_decent(X_train, y_train, w_init, b_init)
print("Optimized weights:", result_w)
print("Optimized bias:", result_b)
