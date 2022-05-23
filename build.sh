#sudo docker rmi -f $(sudo docker images -q)

sudo docker build ./preprocess_data --tag milowb/kubeflow_pipeline_mlflow_preprocessing:latest
sudo docker push milowb/kubeflow_pipeline_mlflow_preprocessing:latest

sudo docker build ./train_model --tag milowb/kubeflow_pipeline_mlflow_train:latest
sudo docker push milowb/kubeflow_pipeline_mlflow_train:latest

sudo docker build ./predict --tag milowb/kubeflow_pipeline_mlflow_predict:latest
sudo docker push milowb/kubeflow_pipeline_mlflow_predict:latest

sudo docker image ls

