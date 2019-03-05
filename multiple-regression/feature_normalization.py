# Feature Normalization
import numpy as np
import pandas as pd


def feature_normalize(x, size):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You need to perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    x_norm = x.copy()
    mu = np.zeros(x.shape[1])
    sigma = np.zeros(x.shape[1])

    features = x.shape[1]
    for feature in range(features):
        mu[feature] = np.mean(x.iloc[:, feature])
        sigma[feature] = np.std(x.iloc[:, feature])
        x_norm.iloc[:, feature] = (x.iloc[:, feature] - mu[feature]) / sigma[feature]

    # now we add the intercept term to housing_data


    intercept = pd.DataFrame(np.ones(size, 1))
    x_norm.join(intercept)
    #  x_norm = np.concatenate([np.ones((size, 1)), x_norm], axis=1)
    # adds row of 1s to first column of X_norm

    return x_norm, mu, sigma

