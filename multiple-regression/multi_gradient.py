import numpy as np

from multi_cost import multi_cost


def gradient_descent(x, y, theta, alpha, iterations):
    """
        Performs gradient descent to optimize the 'theta' parameters. Updates theta for a total of
        inputted 'iterations', with a learning rate 'alpha'.

        Parameters
        ----------
        x : array_like
            Shape (m, n+1), where m is the number of examples, and n+1 the number of features
            including the vector of ones for the zeroth parameter.

        y : array_like
            Shape (m, 1), where m is the value of the function at each point.

        theta : array_like
            Shape (1, n+1). The multiple regression parameters.

        alpha : float
            The learning rate.

        iterations : int
            The number of iterations for gradient descent.

        Returns
        -------
        theta : array_like
            Shape (1, n+1). The learned multiple regression parameters.

        cost_history : list
            A list of the values from the cost function after each iteration.
    """

    # Create temporary array for updating theta
    size = y.shape[0]
    parameters = x.shape[1]
    temp = np.zeros((1, parameters))
    cost_history = np.zeros(iterations)

    for iteration in range(iterations):

        error = (1 / size) * ((x @ theta.T) - y)

        for parameter in range(parameters):
            delta = error * x[:, [parameter]]
            temp[0, parameter] = theta[0, parameter] - (alpha * delta.sum())

        theta = temp
        cost_history[iteration] = multi_cost(x, y, theta)

    return theta, cost_history
