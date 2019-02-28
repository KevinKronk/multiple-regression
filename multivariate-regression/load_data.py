import numpy as np

housing_data = np.loadtxt(os.path.join('/Users/kevkr/Desktop/Data_Folder', 'ex1data2.txt'), delimiter=',')
housing_features = housing_data[:, :2]
housing_prices = housing_data[:, 2]
data_length = housing_prices.size

print('{:>8s}{:>8s}{:>10s}'.format('housing_features[:,0]', 'housing_features[:, 1]', 'housing_prices'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(housing_features[i, 0], housing_features[i, 1], housing_prices[i]))