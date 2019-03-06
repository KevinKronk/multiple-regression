import numpy as np
from multi_cost import multi_cost


def gradient_descent(x, y, theta, alpha, iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    x : array_like
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
    size = y.shape[0]  # number of training examples

    # make a copy of theta, which will be updated by gradient descent
    parameters = x.shape[1]
    temp = np.zeros((1, parameters))
    cost_history = np.zeros(iters)

    for i in range(iters):

        error = (1 / size) * ((np.dot(x, theta.T)) - y)

        for parameter in range(parameters):
            delta = error * x[:, [parameter]]

            temp[0, parameter] = theta[0, parameter] - (alpha * delta.sum())

        # delta2 = delta.sum(axis=1, keepdims=True)
        theta = temp
        cost_history[i] = multi_cost(x, y, theta)  # misses the last theta update

    return theta, cost_history
