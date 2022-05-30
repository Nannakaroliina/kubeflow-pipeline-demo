from feast import FeatureStore
from datetime import datetime, timedelta

# Getting our FeatureStore
store = FeatureStore(repo_path="breast_cancer/")

# Loading the latest features after a previous materialize call or from the beginning of time
store.materialize_incremental(end_date=datetime.now())