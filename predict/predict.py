import joblib
import numpy as np
from sklearn.metrics import f1_score
import argparse


def predict(x_test, y_test, model):
    x_test_ = np.load(x_test)
    y_test_ = np.load(y_test)

    model = joblib.load(model)

    y_pred = model.predict(x_test_)

    score = f1_score(y_test_, y_pred, average='micro')

    with open('results.txt', 'w') as result:
        result.write(f'Score: {score}  \n\n Prediciton: {y_pred} | Actual {y_test_}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    predict(args.x_test, args.y_test, args.model)