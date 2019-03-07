import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data
from multi_cost import multi_cost
from multi_gradient import gradient_descent
from prediction import predict_price


# Linear regression with multiple variables - predicting housing prices


# Load Data

filename = 'housing_data.txt'
housing_data = load_data(filename)


# Feature Normalization

# store the means and standard deviations for predictions
means = housing_data.mean()
stds = housing_data.std()

housing_data = (housing_data - housing_data.mean()) / housing_data.std()


# Inserting intercept parameter

housing_data.insert(0, 'Ones', 1)


# Convert pandas DataFrame to numpy arrays

cols = housing_data.shape[1]

x = housing_data.iloc[:, 0:cols-1]
y = housing_data.iloc[:, cols-1:cols]

x = x.values
y = y.values


# Set initial values

alpha = 0.03
iterations = 1000
theta = np.array([[0, 0, 0]])


# Cost Function

cost = multi_cost(x, y, theta)


# Gradient Descent

new_theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)


# Plotting Data

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(range(iterations)), cost_history, 'r')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History in Epochs")
plt.show()


# Prediction

sq_feet = 1650
bedrooms = 3
new_price = predict_price(sq_feet, bedrooms, new_theta, means, stds)
print(f"The price for a {bedrooms} bedroom, {sq_feet} sq foot house is:\n\t${new_price}")
