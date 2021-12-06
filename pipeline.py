import kfp
from kfp.v2 import dsl


def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='Nannakaroliina/kubeflow_pipeline_demo_preprocessing:latest',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )


def train_op(x_train, y_train):

    return dsl.ContainerOp(
        name='Train Model',
        image='Nannakaroliina/kubeflow_pipeline_demo_train:latest',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )


def predict_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='Nannakaroliina/kubeflow_pipeline_demo_predict:latest',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'mean_squared_error': '/app/output.txt'
        }
    )


@dsl.pipeline(
    name='Kubeflow pipeline demo',
    description='Kubeflow pipeline demo with simple model training'
)
def kubeflow_demo_pipeline():
    _preprocess_op = preprocess_op()

    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)

    _test_op = predict_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)


client = kfp.Client(host="http://127.0.0.1:8080/pipeline")
client.create_run_from_pipeline_func(kubeflow_demo_pipeline, arguments={})