import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from joblib import dump
import shutil

store = FeatureStore(repo_path="breast_cancer/")
training_df = store.get_saved_dataset(name="breast_cancer_dataset").to_df()
labels = training_df['label']
features = training_df.drop(
    labels=['label', 'event_timestamp', "patient_id"], 
    axis=1)

# Splitting the dataset into train and test sets 
# use stratify to make sure the proportion of values in classes (a% label 0 and b* label 1)
X_train, X_test, y_train, y_test = train_test_split(features, 
                                                    labels, 
                                                    stratify=labels) 
reg = LogisticRegression()
reg.fit(X=X_train[sorted(X_train)], y=y_train)

# Saving the model
dump(reg, filename="model.joblib")
shutil.make_archive('feast_dir', 'zip', 'breast_cancer')