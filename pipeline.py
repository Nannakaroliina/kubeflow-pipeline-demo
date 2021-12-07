import kfp
from kfp import dsl


# Build ContainerOps of Docker images with needed arguments for data/model access and file_output definitions.
# TODO Fix the FutureWarning regarding ContainerOp

def preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='nannakaroliina/kubeflow_pipeline_demo_preprocessing:latest',
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
        image='nannakaroliina/kubeflow_pipeline_demo_train:latest',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.plk'
        }
    )


def predict_op(x_test, y_test, model):
    return dsl.ContainerOp(
        name='Test Model',
        image='nannakaroliina/kubeflow_pipeline_demo_predict:latest',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'f1_score': '/app/results.txt'
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


if __name__ == '__main__':
    # Build a pipeline yaml file to be uploaded to Kubeflow Pipeline UI
    # TODO implement local run option without manual pipeline creation
    kfp.compiler.Compiler().compile(kubeflow_demo_pipeline, 'pipeline.yaml')
