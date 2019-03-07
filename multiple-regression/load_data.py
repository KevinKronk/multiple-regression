import pandas as pd


def load_data(filename):
    """ Loads a csv file with one or more features into a DataFrame. """

    df = pd.read_csv(filename)

    return df
