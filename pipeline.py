from xml.sax.handler import DTDHandler
import kfp
from kfp import dsl
import kfp.compiler as compiler
import json

# Build ContainerOps of Docker images with needed arguments for data/model access and file_output definitions.
# TODO Fix the FutureWarning regarding ContainerOp

def preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='milowb/kubeflow_pipeline_mlflow_preprocessing:latest',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy'
        }
    )


def train_op(x_train, y_train, model_name):
    return dsl.ContainerOp(
        name='Train Model',
        image="milowb/kubeflow_pipeline_mlflow_train:latest",
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train,
            "--model_name", model_name
        ],
        file_outputs={
            'model': '/app/model.plk'
        }
    )


def predict_op(x_test, y_test, model, model_name):
    return dsl.ContainerOp(
        name='Test Model',
        image='milowb/kubeflow_pipeline_mlflow_predict:latest',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model,
            '--model_name', model_name
        ],
        file_outputs={
            'f1_score': '/app/results.txt'
        }
    )


@dsl.pipeline(
    name='Kubeflow pipeline and mlflow demo',
    description='Kubeflow pipeline and mlflow demo with simple model training'
)
def kubeflow_mlflow_pipeline():
    _preprocess_op = preprocess_op()

    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train']),
        "model_name_XXX3"
    ).after(_preprocess_op)

    _test_op = predict_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model']),
        "model_name_XXX3"
    ).after(_train_op)


if __name__ == '__main__':
    
    EXPERIMENT_NAME="Potatoe"

    pipeline_func = kubeflow_mlflow_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.zip'

    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    client = kfp.Client(host='http://127.0.0.1:8080/') # change arguments accordingly    
    experiment = None
    try:
        experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
    except:
        experiment = client.create_experiment(EXPERIMENT_NAME)
    
    #print(client.list_experiments())
    #print("client.pipelines.list_pipelines(filter=filter)", client.pipelines.list_pipelines())

    arguments = {}

    """
    runs = client.list_runs().runs
    for run in runs:
        print(run.id)
        run.terminate_run()
    """

    run_name = pipeline_func.__name__ + ' run'
    run_result = client.run_pipeline(experiment.id, 
                                    run_name, 
                                    pipeline_filename, 
                                    arguments)

    completed_run = client.wait_for_run_completion(run_id=run_result.id, timeout=float("inf"))
    print(completed_run)


    """
    result = client.create_run_from_pipeline_func(
        kubeflow_mlflow_pipeline,
        enable_caching=False,
        # You can optionally override your pipeline_root when submitting the run too:
        # pipeline_root='gs://my-pipeline-root/example-pipeline',
        arguments = {
        }
    )










    client.run_pipeline(experiment.id, "test")
    

    run_name = kubeflow_mlflow_pipeline.__name__ + ' run'
    run_result = client.run_pipeline(experiment.id, 
                                    run_name, 
                                    pipeline_filename, 
                                    arguments)
    """

    # Build a pipeline yaml file to be uploaded to Kubeflow Pipeline UI
    # TODO implement local run option without manual pipeline creation
    # kfp.compiler.Compiler().compile(kubeflow_mlflow_pipeline, 'pipeline.yaml')
    
