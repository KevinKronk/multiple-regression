import pandas as pd


def load_data(filename):
    """ Loads a csv file with two or more columns. """

    df = pd.read_csv(filename)
    x = df.iloc[:, [0, 1]]
    y = df.iloc[:, 2]
    size = y.size

    return x, y, size

