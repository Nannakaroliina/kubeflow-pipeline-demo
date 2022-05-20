import argparse
import mlflow
import pickle
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

model_name = "model.plk"


def _train_model(x_train, y_train):
    x_train_ = np.load(x_train)
    y_train_ = np.load(y_train)

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
    grid.fit(x_train_, y_train_)

    pickle.dump(grid, open(model_name, 'wb'))
    # joblib.dump(grid, model_name)


if __name__ == '__main__':
    #model = pickle.load(open(model_name, 'rb'))
    #mlflow.set_tracking_uri("http://127.0.0.1:8000")
    print("COCOA")
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    _train_model(args.x_train, args.y_train)
    """
    mlflow.set_tracking_uri("file:///tmp/my_tracking")
    mlflow.set_experiment("train-model")

    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    _train_model(args.x_train, args.y_train)

    model = pickle.load(open(model_name, 'rb'))

    print("--")
    mlflow.sklearn.log_model(model, "sk_learn",
                             serialization_format="cloudpickle",
                             registered_model_name="sklearn-grid-search-cv-model")
    """