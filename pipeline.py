import kfp
from kfp import dsl


def train_op():
    return dsl.ContainerOp(
        name='Train Model',
        image='hmnguyen/train:latest',
        file_outputs={
            'x_train': 'app/cifar_trX.bin',
            'y_train': 'app/cifra_trY.bin',
            'x_test': 'app/cifar_tsX.bin',
            'y_test': 'app/cifar_tsY.bin',
            'architecture': 'app/model.pdf',
            'model': '/app/simple_CNN.bin'
        }
    )

    
def predict_op(x_test, y_test, model):
    return dsl.ContainerOp(
        name='Test Model',
        image='hmnguyen/predict:latest',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ]
    )


@dsl.pipeline(
    name='Kubeflow pipeline for EDDL demo',
    description='Kubeflow pipeline demo with simple EDDL model training'
)

def kubeflow_eddl_pipeline():
    _train_op = train_op()

    _test_op = predict_op(
        dsl.InputArgumentPath(_train_op.outputs['x_test']),
        dsl.InputArgumentPath(_train_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)


if __name__ == '__main__':
    # Build a pipeline yaml file to be uploaded to Kubeflow Pipeline UI
    # TODO implement local run option without manual pipeline creation
    kfp.compiler.Compiler().compile(kubeflow_eddl_pipeline, 'pipeline.yaml')

