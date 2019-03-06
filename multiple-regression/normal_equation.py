import numpy as np
from load_data import load_data


filename = 'housing_data.txt'
housing_data = load_data(filename)

housing_data.insert(0, 'Ones', 1)
cols = housing_data.shape[1]
x = housing_data.iloc[:, 0:cols-1]
y = housing_data.iloc[:, cols-1:cols]  # why do I need the :cols

x = x.values
y = y.values


def normal_equation(x, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    x : array_like
        The dataset of shape (m x n+1).

    y : array_like
        The value at each data point. A vector of shape (m, ).

    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).

    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    """
    # theta = np.zeros(X.shape[1])

    norm_theta = (np.linalg.pinv(x.T @ x) @ x.T) @ y
    return norm_theta


theta = normal_equation(x, y)
print(theta)

new_house = np.array([1, 1650, 3])
new_price = new_house @ theta
print(new_price)
