# Feature definition file
from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Feature, FileSource, ValueType
from feast.types import Float32, Int64

# Define an patient entity
patient = Entity(name="patient_id", value_type=ValueType.INT64, description="ID of patient")

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
data_src = FileSource(
    path="breast_cancer/data/data.parquet",
    timestamp_field="event_timestamp",
    event_timestamp_column="event_timestamp",
)

label_src = FileSource(
    path="breast_cancer/data/label.parquet", 
    timestamp_field="event_timestamp",
    event_timestamp_column="event_timestamp",
)

# Define a Feature View that will allow us to serve this data to our model online.
data_fv = FeatureView(
    name="breast_feature_view",
    entities=["patient_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="mean radius", dtype=ValueType.FLOAT),
        Feature(name="mean texture", dtype=ValueType.FLOAT),
        Feature(name="mean perimeter", dtype=ValueType.FLOAT),
        Feature(name="mean area", dtype=ValueType.FLOAT),
        Feature(name="mean smoothness", dtype=ValueType.FLOAT),
        Feature(name="mean compactness", dtype=ValueType.FLOAT),
        Feature(name="mean concavity", dtype=ValueType.FLOAT),
        Feature(name="mean concave points", dtype=ValueType.FLOAT),
        Feature(name="mean symmetry", dtype=ValueType.FLOAT),
        Feature(name="mean fractal dimension", dtype=ValueType.FLOAT),
        Feature(name="radius error", dtype=ValueType.FLOAT),
        Feature(name="texture error", dtype=ValueType.FLOAT),
        Feature(name="perimeter error", dtype=ValueType.FLOAT),
        Feature(name="area error", dtype=ValueType.FLOAT),
        Feature(name="smoothness error", dtype=ValueType.FLOAT),
        Feature(name="compactness error", dtype=ValueType.FLOAT),
        Feature(name="concavity error", dtype=ValueType.FLOAT),
        Feature(name="concave points error", dtype=ValueType.FLOAT),
        Feature(name="symmetry error", dtype=ValueType.FLOAT),
        Feature(name="fractal dimension error", dtype=ValueType.FLOAT),
        Feature(name="worst radius", dtype=ValueType.FLOAT),
        Feature(name="worst texture", dtype=ValueType.FLOAT),
        Feature(name="worst perimeter", dtype=ValueType.FLOAT),
        Feature(name="worst area", dtype=ValueType.FLOAT),
        Feature(name="worst smoothness", dtype=ValueType.FLOAT),
        Feature(name="worst compactness", dtype=ValueType.FLOAT),
        Feature(name="worst concavity", dtype=ValueType.FLOAT),
        Feature(name="worst concave points", dtype=ValueType.FLOAT),
        Feature(name="worst symmetry", dtype=ValueType.FLOAT),
        Feature(name="worst fractal dimension", dtype=ValueType.FLOAT)
    ],
    source=data_src
)

label_fv = FeatureView(
    name="label_feature_view",
    entities=["patient_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="label", dtype=ValueType.INT32)        
        ],    
    batch_source=label_src
)