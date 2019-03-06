import numpy as np


def predict_price(sq_feet, bedrooms, means, stds, theta):

    p_feet = (sq_feet - means['sq_feet']) / stds['sq_feet']
    p_bedrooms = (bedrooms - means['bedrooms']) / stds['bedrooms']

    prediction = np.array([1, p_feet, p_bedrooms])

    newprice = np.dot(prediction, theta.T)
    price = newprice * stds['price'] + means['price']
    return price