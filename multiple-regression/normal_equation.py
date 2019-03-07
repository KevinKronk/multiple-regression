import numpy as np

from load_data import load_data


# Using the normal equation to solve for the global min theta parameters


# Load data

filename = 'housing_data.txt'
housing_data = load_data(filename)


# Insert intercept parameter

housing_data.insert(0, 'Ones', 1)


# Convert pandas DataFrame to numpy arrays

cols = housing_data.shape[1]

x_df = housing_data.iloc[:, 0:cols-1]
y_df = housing_data.iloc[:, cols-1:cols]

x_array = x_df.values
y_array = y_df.values


def normal_equation(x, y):
    """
        Computes the closed-form solution to multiple regression using the normal equation.

        Parameters
        ----------
        x : array_like
            Shape (m, n+1), where m is the number of examples, and n+1 the number of features
            including the vector of ones for the zeroth parameter.

        y : array_like
            Shape (m, 1), where m is the value of the function at each point.

        Returns
        -------
        normal_theta : array_like
            Shape (n+1, 1). Solved multiple regression parameters.
    """

    normal_theta = (np.linalg.pinv(x.T @ x) @ x.T) @ y
    return normal_theta


# Get theta parameters from the normal equation

theta = normal_equation(x_array, y_array)


# Predict house price with theta parameters

sq_feet = 1650
bedrooms = 3
new_house = np.array([1, sq_feet, bedrooms])
new_price = int(new_house @ theta)

print(f"The price for a {bedrooms} bedroom, {sq_feet} sq foot house is:\n\t${new_price}")
