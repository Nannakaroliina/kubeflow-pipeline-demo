import kfp
from kfp import dsl


# Build ContainerOps of Docker images with needed arguments for data/model access and file_output definitions.
# TODO Fix the FutureWarning regarding ContainerOp

def train_op():
    return dsl.ContainerOp(
        name='Train Model',
        image='nannakaroliina/feasttrain:latest',
        file_outputs={
            'model': '/app/model.joblib',
            'featurestore': '/app/feast_dir.zip'
        }
    )


def predict_op(featurestore, model):
    return dsl.ContainerOp(
        name='Test Model',
        image='nannakaroliina/feastpredict:latest',
        arguments=[
            '--featurestore', featurestore,
            '--model', model
        ]
    )


@dsl.pipeline(
    name='Kubeflow Feast demo',
    description='Kubeflow pipeline demo with Feast'
)
def kubeflow_feast_pipeline():
    _train_op = train_op()

    _test_op = predict_op(
        dsl.InputArgumentPath(_train_op.outputs['featurestore']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)


if __name__ == '__main__':
    # Build a pipeline yaml file to be uploaded to Kubeflow Pipeline UI
    # TODO implement local run option without manual pipeline creation
    kfp.compiler.Compiler().compile(kubeflow_feast_pipeline, 'pipeline.yaml')
