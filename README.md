# Kubeflow pipeline demo
Kubeflow pipeline demo for MLOps purposes.
This project contains components _(see photo below)_ to preprocess data for ML model,
train ML model and testing/using the model to predict. Each component is
separate Docker instance that are build from the recent code using Github workflow action.

Preprocessing component takes a dataset directly from sklearn and
splits the data to x and y, train and test parts. X_train and x_test are optimised for more accurate results.
Data partitions are saved using numpy to _/app_ directory. 
Train component loads the saved data through arguments given for the _ContainerOp_ and uses it to
train a model. Model is saved to the _/app_ directory as _model.plk_.
Testing components loads the saved data and model through the arguments
and gives the prediction and score for the prediction.

Pipeline is build with _kfp_ package which is Kubeflow Pipeline SDK API.
Creating each individual op implementation of a container image with _ContainerOp_. 
Images are pulled from Docker for each component. For each op the needed _arguments_ and
_file_outputs_ are defined on _pipeline.py_. Pipeline flow is defined in _kubeflow_demo_pipeline_ and 
the _pipeline.yaml_ file is compiled based on the _pipeline.py_ code.

MLflow integration to the pipeline adds the functionality for model saving to desired backend storage,
in this case PostgreSQL, model versioning, logging of metrics etc. See more at: [MLflow documentation](https://mlflow.org/docs/latest/index.html)


![alt text](photos/kubeflow-pipline.png "Kubeflow pipeline components")

### Code modifications

In case of code modifications to _pipeline.py_, run the code to generate new _pipeline.yaml_.
If modifying the components, remember to push changes to Github to get newest version for the Docker images.
While using new branches for new implementations, the docker images are not built so this requires bit of manual
work unless the code is merged to main. I haven't added action workflow for building docker images for branches.

## Requirements

* Kubernetes
* Kubeflow (kind, K3s, K3ai)
* MLflow (local unless server available)
* PostgreSQL

## Installation steps for local Kubeflow Pipelines
There is three options: kind, K3s or K3ai. 
Following steps are for K3s, so for other options please visit: 
[Kubeflow Documentation, Local Deployment](https://www.kubeflow.org/docs/components/pipelines/installation/localcluster-deployment/)

### K3s

Uninstall Previous Versions of K3s
```shell
sh /usr/local/bin/k3s-uninstall.sh
```

### Install K3s:
```shell
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.21.8+k3s1 sh -
```
#### NOTE
The version after 1.21 does not support for some APIs of K3s in kubeflow. 
More information at [link](https://stackoverflow.com/questions/70420241/issues-when-setting-up-kubeflow-pipeline-on-minikube/70450847)


### Create a cluster:
```shell
sudo k3s server &
```

### Check the cluster exists:
```shell
sudo k3s kubectl get node
```

### Deploying Kubeflow Pipelines
Following installation works for all environments (kind, K3s, K3ai).

Deploy the Kubeflow Pipelines
```shell
export PIPELINE_VERSION=1.7.1

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

Verify that the UI is accessible by port forwarding it:
```shell
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

You should see this when port-forward is successful.

![alt text](photos/port-forwarding.png "Successful port-forwarding")

Now you can access the UI at [http:/localhost:8080/](http://localhost:8080/).
In any case of issues check installation guide above or ask for further assistance.


### Using the Kubeflow Pipelines UI

After opening the UI _(see below)_, click the _+ Upload pipeline_ to create new pipeline.

![alt text](photos/kubeflow-ui-home.png "Kubeflow Pipelines UI")

Set name, description and upload _pipeline.yaml_ file for the new pipeline and click _create_.

![alt text](photos/upload-pipeline.png "Upload pipeline yaml")

This will lead to summary view, click _+ Create run_ button from top of the window.
Run details contains everything needed for basic run, click _Start_.

![alt text](photos/start-run.png "Start new run")

Finally the run view will open and the pipeline is running. 
After run the results will be available on the side panel.

## Setting up the local MLflow

For the backend storage, one desired database needs to be available. 
In this demo, PostgreSQL is chosen database and it can be replaces as you desire.

### Setup the PostgreSQL

If you don't have postgres installed, install it before next step. Find the correct version from
[PostgreSQL downloads](https://www.postgresql.org/download/)

Start postgres:
```shell
sudo -u postgres psql
```

Create database for MLflow with user and password _(if you change the name of db, user or password
remember to modify them in next command as well)_:
```postgresql
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
```

PostgreSQL needs psycopg2 library _(included in requirements)_ for Python to interact with it. To ensure correct installation make sure you have GCC Linux installed:
```shell
sudo apt install gcc
```

Since PostgreSQL is meant for metadata storing, we need local folders for runs and logs:
```shell
mkdir ~/mlflow/mlruns
mkdir ~/mlflow/mllogs
```

### Start the local MLflow Tracking service
To run the MLflow locally, use following command:
```shell
mlflow server \
  --backend-store-uri postgresql://mlflow_user:password@localhost/mlflow_db \
  --default-artifact-root sftp://mlflow_user@<hostname_of_server>:~/mlflow/mlruns \
  -h 0.0.0.0 \
  -p 8000
```

For setting up the remote server for MLflow, you can find more information from [MLflow documentation](https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers) and
there is articles available as well by other developers.

