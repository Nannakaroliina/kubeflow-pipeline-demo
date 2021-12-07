import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def _preprocess_data():
    # TODO Should be able to use different datasets, shouldn't be hardcoded one
    cancer = load_breast_cancer()
    df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

    x = df.drop(['target'], axis=1)
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
    x_train, x_test = optimize(x_train, x_test)

    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)


def optimize(x_train, x_test):
    min_train = x_train.min()
    range_train = (x_train - min_train).max()
    x_train_scaled = (x_train - min_train) / range_train

    min_test = x_test.min()
    range_test = (x_test - min_test).max()
    x_test_scaled = (x_test - min_test) / range_test
    return x_train_scaled, x_test_scaled


if __name__ == '__main__':
    print('Preprocessing data...')

    _preprocess_data()
