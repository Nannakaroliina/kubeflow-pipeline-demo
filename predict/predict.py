from feast import FeatureStore
import pandas as pd
from joblib import load
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--featurestore')
parser.add_argument('--model')
args = parser.parse_args()

shutil.unpack_archive(args.featurestore, 'breast_cancer', 'zip')
store = FeatureStore(repo_path="breast_cancer/")
infer_features = [
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
test_data = store.get_online_features(
    features=infer_features,    
    entity_rows=[{"patient_id": 568}, {"patient_id": 567}]
).to_dict()
print(test_data)
test_df = pd.DataFrame.from_dict(data=test_data).dropna()
print(test_df)
if len(df):
    reg = load(args.model)
    predictions = reg.predict(
        test_df[sorted(test_df.drop("patient_id", axis=1))])
    print(predictions)