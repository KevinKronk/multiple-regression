import numpy as np


def multi_cost(x, y, theta):
    """
        Compute multiple regression cost for multiple features with given theta parameter values.

        Parameters
        ----------
        x : array_like
            Shape (m, n+1), where m is the number of examples, and n+1 the number of features
            including the vector of ones for the zeroth parameter.

        y : array_like
            Shape (m, 1), where m is the value of the function at each point.

        theta : array_like
            Shape (1, n+1). The multiple regression parameters.

        Returns
        -------
        cost : float
            The value of the regression cost function.
    """

    size = y.shape[0]

    cost = np.sum((1 / (2 * size)) * (((x @ theta.T) - y) ** 2))
    return cost
