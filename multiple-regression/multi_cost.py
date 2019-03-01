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