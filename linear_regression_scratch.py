import numpy as np
from matplotlib import pyplot as plt

def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    total_cost = 0
    for i in range(m):
        Fw_b = w * x_train[i] + b
        total_cost += (Fw_b - y_train[i]) ** 2
    return total_cost / (2 * m)

def compute_gradient(x, y, w, b, alpha=0.1):
    m = x.shape[0]
    total_gradient_w = 0
    total_gradient_b = 0
    for i in range(m):
        Fw_b = w * x[i] + b
        total_gradient_b += (Fw_b - y[i])
        total_gradient_w += (Fw_b - y[i]) * x[i]
    return (alpha * total_gradient_w) / m, (alpha * total_gradient_b) / m

def gradient_descent(x, y, w, b, alpha=0.1, number_of_iterations=10000):
    cost_list = []
    w_list = []
    for i in range(number_of_iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= dj_dw
        b -= dj_db
        w_list.append(w)
        cost_list.append(compute_cost(x, y, w, b))
    return w, b, w_list, cost_list

def predict_line(x, w, b):
    return w * x + b

# Given data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([14, 15, 16, 17, 19])

# Initial parameters
initial_w = 0.5
initial_b = 1

# Perform gradient descent
w, b, w_list, cost_list = gradient_descent(x, y, initial_w, initial_b)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Subplot 1: Scatter plot of original points
axs[0, 0].scatter(x, y, color='blue')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('Scatter Plot of Original Points')

# Subplot 2: Original points and predicted line
axs[0, 1].scatter(x, y, label='Original Points', color='blue')
axs[0, 1].plot(x, predict_line(x, w, b), label='Predicted Line', color='red')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[0, 1].legend()
axs[0, 1].set_title('Original Points and Predicted Line')

# Subplot 3: Cost vs. w
axs[1, 0].plot(w_list, cost_list, color='green')
axs[1, 0].set_xlabel('W')
axs[1, 0].set_ylabel('Cost Function')
axs[1, 0].set_title('Cost vs. W')

# Subplot 4: Cost vs. Number of Iterations
axs[1, 1].plot(range(10000), cost_list, color='purple')
axs[1, 1].set_xlabel('Number of Iterations')
axs[1, 1].set_ylabel('Cost Function')
axs[1, 1].set_title('Cost vs. Number of Iterations')

# Display the plots
plt.tight_layout()
plt.show()

print(f'For x = 6, y = {predict_line(6, w, b)}')
