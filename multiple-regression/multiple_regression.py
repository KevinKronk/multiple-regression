import numpy as np
import pandas as pd
from load_data import load_data
import matplotlib.pyplot as plt
from multi_cost import multi_cost
from multi_gradient import gradient_descent
from prediction import predict_price


# Linear regression with multiple variables - predicting housing prices


# Load Data

filename = 'housing_data.txt'
housing_data = load_data(filename)

alpha = 0.01
iterations = 1000

# Feature Normalization
means = housing_data.mean()
stds = housing_data.std()

housing_data = (housing_data - housing_data.mean()) / housing_data.std()


# Inserting Intercept parameter

housing_data.insert(0, 'Ones', 1)


# Converting Pandas dataframe to necessary numpy arrays

cols = housing_data.shape[1]

x = housing_data.iloc[:, 0:cols-1]
y = housing_data.iloc[:, cols-1:cols]  # why do I need the :cols

x = x.values
y = y.values
theta = np.array([[0, 0, 0]])


cost = multi_cost(x, y, theta)


theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)
print(theta)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(range(iterations)), cost_history, 'r')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost History in Epochs")
plt.show()


# Prediction

pprice = predict_price(1650, 3, means, stds, theta)
print(pprice)

'''


# make a prediction for a 1650 sq-ft, 3 br house
f1 = 1650
f2 = 3
f3 = (np.dot([1, f1, f2], new_theta2))
predict3 = float(f3*sigma[2])+mu[2]
print("Predicted price of a 1650 sq-ft, 3 bed room house: ${:.0f}".format(predict3))

'''