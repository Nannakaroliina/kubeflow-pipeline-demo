import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

store = FeatureStore(repo_path='breast_cancer/')
entity_df = pd.read_parquet(path="breast_cancer/data/label.parquet")  

data_retrieval = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "breast_feature_view:mean radius",
        "breast_feature_view:mean texture",
        "breast_feature_view:mean perimeter",
        "breast_feature_view:mean area",
        "breast_feature_view:mean smoothness",
        "breast_feature_view:mean compactness",
        "breast_feature_view:mean concavity",
        "breast_feature_view:mean concave points",
        "breast_feature_view:mean symmetry",
        "breast_feature_view:mean fractal dimension",
        "breast_feature_view:radius error",
        "breast_feature_view:texture error",
        "breast_feature_view:perimeter error",
        "breast_feature_view:area error",
        "breast_feature_view:smoothness error",
        "breast_feature_view:compactness error",
        "breast_feature_view:concavity error",
        "breast_feature_view:concave points error",
        "breast_feature_view:symmetry error",
        "breast_feature_view:fractal dimension error",
        "breast_feature_view:worst radius",
        "breast_feature_view:worst texture",
        "breast_feature_view:worst perimeter",
        "breast_feature_view:worst area",
        "breast_feature_view:worst smoothness",
        "breast_feature_view:worst compactness",
        "breast_feature_view:worst concavity",
        "breast_feature_view:worst concave points",
        "breast_feature_view:worst symmetry",
        "breast_feature_view:worst fractal dimension"
    ]
)

dataset = store.create_saved_dataset(
    from_=data_retrieval,
    name="breast_cancer_dataset",
    storage=SavedDatasetFileStorage("breast_cancer/data/breast_cancer_dataset.parquet")
)