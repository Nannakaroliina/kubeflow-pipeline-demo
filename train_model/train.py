import argparse
import mlflow
from mlflow.tracking import MlflowClient
import pickle
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

def print_model_info(rm):
    print("--")
    print("name: {}".format(rm.name))
    print("tags: {}".format(rm.tags))
    print("description: {}".format(rm.description))

if __name__ == '__main__':
    print("COCOA3")
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()


    _train_model(args.x_train, args.y_train)
    model = pickle.load(open(model_name, 'rb'))
    model_name = "sklearn-grid-search-cv-model3"
    mlflow.set_tracking_uri("http://172.17.0.2:5000")
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "sk_learn",
                             serialization_format="cloudpickle",
                             registered_model_name=model_name)
    client = MlflowClient()
    model = client.get_registered_model(model_name)

    # It shows some parameters of the model
    print_model_info(model)
    desc = "This sentiment analysis model classifies tweets' tone: happy, sad, angry."
    client.update_registered_model(model_name, desc)
    model = client.get_registered_model(model_name)
    # It shows the same model with an updated description
    print_model_info(model)