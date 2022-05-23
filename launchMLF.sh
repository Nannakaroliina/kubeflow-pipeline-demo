sudo docker rm $(docker ps -aq --filter name=atcommons/mlflow-server)
sudo docker run -it --rm -p 5000:5000 -v /local/path:/mlflow --name mlflow-server atcommons/mlflow-server