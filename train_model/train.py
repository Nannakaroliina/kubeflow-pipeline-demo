import argparse

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def _train_model(x_train, y_train):
    x_train_ = np.load(x_train)
    y_train_ = np.load(y_train)

    model_name = "model.plk"

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
    grid.fit(x_train_, y_train_)

    joblib.dump(grid, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    _train_model(args.x_train, args.y_train)
