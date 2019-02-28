# Linear regression with multiple variables - predicting housing prices


# Cost Function for multiple variables
housing_features = housing_data[:, :3]
housing_prices = housing_data[:, 3]
housing_prices = housing_prices[:, None]
data_length = housing_prices.size

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    Returns
    -------
    J : float
        The value of the cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    m = y.shape[0]  # number of training examples

    first = (np.dot(X, theta) - y).T
    second = (np.dot(X, theta) - y)
    cost = np.sum((1/(2*m))*(np.dot(first, second)))
    return cost


multi_J = computeCostMulti(housing_features, housing_prices, theta=np.array([[0], [0], [0]]))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    alpha : float
        The learning rate for gradient descent.

    num_iters : int
        The number of iterations to run gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Perform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()

    J_history = []

    for i in range(num_iters):
        temp_J = computeCostMulti(X, y, theta)
        J_history.append(temp_J)
        delta = (1/m)*((np.dot(X, theta))-y)*X
        delta2 = np.array(delta.sum(axis=0))
        delta3 = delta2.reshape(-1, 1)
        theta = (theta - (alpha*delta3))
        i += 1
    return theta, J_history


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