# reloading data
housing_data = np.loadtxt(os.path.join('/Users/kevkr/Desktop/Data_Folder', 'ex1data2.txt'), delimiter=',')
housing_features = housing_data[:, :2]
housing_prices = housing_data[:, 2]
housing_prices = np.reshape(housing_prices, (-1, 1))
data_length = housing_prices.size
housing_features = np.concatenate([np.ones((data_length, 1)), housing_features], axis=1)


def normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    X : array_like
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

    norm_theta = (np.linalg.pinv(X.T @ X) @ X.T) @ y
    return norm_theta


normal_theta = normalEqn(housing_features, housing_prices)
print(normal_theta)

predict4 = float(np.dot([1, 1650, 3], normal_theta))
print("Predicted price of a 1650 sq-ft, 3 bed room house: ${:.0f}".format(predict4))