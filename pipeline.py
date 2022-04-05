import kfp
from kfp import dsl


def train_op():
    return dsl.ContainerOp(
        name='Train Model',
        image='nannakaroliina/eddltrain:latest',
        file_outputs={
            'x_train': 'app/cifar_trX.bin',
            'y_train': 'app/cifar_trY.bin',
            'x_test': 'app/cifar_tsX.bin',
            'y_test': 'app/cifar_tsY.bin',
            'architecture': 'app/model.pdf',
            'model': '/app/simple_CNN.bin'
        }
    )

    
def predict_op(model, x_test, y_test):
    return dsl.ContainerOp(
        name='Test Model',
        image='nannakaroliina/eddlpredict:latest',
        arguments=[
            model,
            x_test,
            y_test
        ]
    )


@dsl.pipeline(
    name='Kubeflow pipeline for EDDL demo',
    description='Kubeflow pipeline demo with simple EDDL model training'
)

def kubeflow_eddl_pipeline():
    _train_op = train_op()

    _test_op = predict_op(
        dsl.InputArgumentPath(_train_op.outputs['model']),
        dsl.InputArgumentPath(_train_op.outputs['x_test']),
        dsl.InputArgumentPath(_train_op.outputs['y_test'])
    ).after(_train_op)


if __name__ == '__main__':
    # Build a pipeline yaml file to be uploaded to Kubeflow Pipeline UI
    # TODO implement local run option without manual pipeline creation
    kfp.compiler.Compiler().compile(kubeflow_eddl_pipeline, 'pipeline.yaml')

