# env/platform-agnostic-pns hasn't been publically released, so you will install it from master
export PIPELINE_VERSION=1.8.1
sudo kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
sudo kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
sudo kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
sudo kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
