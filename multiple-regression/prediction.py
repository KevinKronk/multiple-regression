import numpy as np


def predict_price(sq_feet, bedrooms, theta, means, stds):
    """
        Predicts the price of a house with given square feet and bedrooms using the given theta
        parameters. Uses the means and standard deviations to undo feature normalization.

        Parameters
        ----------
        sq_feet : int
            Square feet of the house.

        bedrooms : int
            Number of bedrooms in the house.

        theta : array_like
            Shape (1, n+1). The multiple regression parameters.

        means : array_like
            Shape (n+1,). DataFrame of the means for the features and prices.

        stds : array_like
            Shape (n+1,). DataFrame of the standard deviations for the features and prices.

        Returns
        -------
        price : int
            The predicted price of the house.
    """

    # Feature normalize sq_feet and bedrooms to calculate the new price
    norm_feet = (sq_feet - means['sq_feet']) / stds['sq_feet']
    norm_bedrooms = (bedrooms - means['bedrooms']) / stds['bedrooms']

    prediction = np.array([1, norm_feet, norm_bedrooms])

    norm_price = prediction @ theta.T
    price = int(norm_price * stds['price'] + means['price'])

    return price
