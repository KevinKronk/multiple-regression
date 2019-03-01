# import numpy as np
# import pandas as pd
from load_data import load_data


# Linear regression with multiple variables - predicting housing prices


# Load Data

filename = 'housing_data.txt'
x, y, size = load_data(filename)


'''
X_norm, mu, sigma = featureNormalize(housing_data)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# now we add the intercept term to housing_data
housing_data = np.concatenate([np.ones((data_length, 1)), X_norm], axis=1)
# adds row of 1s to first column of X_norm

multi_J = computeCostMulti(housing_features, housing_prices, theta=np.array([[0], [0], [0]]))



theta = np.array([[0], [0], [0]])
alpha_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 1.1, 1.2]
num_iters = 50
plt.figure(2)

for alpha2 in alpha_list:
    new_theta2, J_history2 = gradientDescentMulti(housing_features, housing_prices, theta, alpha2, num_iters)

    plt.plot(np.arange(len(J_history2)), J_history2, lw=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

# Plot the convergence graph
plt.show(2)

alpha = 1.2
new_theta2, J_history2 = gradientDescentMulti(housing_features, housing_prices, theta, alpha, num_iters)
print(new_theta2)

# make a prediction for a 1650 sq-ft, 3 br house
f1 = (1650-mu[0])/sigma[0]
f2 = (3-mu[1])/sigma[1]
f3 = (np.dot([1, f1, f2], new_theta2))
predict3 = float(f3*sigma[2])+mu[2]
print("Predicted price of a 1650 sq-ft, 3 bed room house: ${:.0f}".format(predict3))

'''