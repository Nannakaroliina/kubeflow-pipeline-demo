import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def _data2parquet():
    # load breast cancer data
    data = datasets.load_breast_cancer()
    # put it into a dataframe
    data_df = pd.DataFrame(data=data.data, columns=data.feature_names)
    label_df = pd.DataFrame(data=data.target, columns=['label'])
    # Creating timestamps and patient ID for data
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(data_df), freq='D').to_frame(name="event_timestamp", index=False)
    patient_ids = pd.DataFrame(data=list(range(len(data_df))), columns=["patient_id"])
    # Normalize the data
    data_df = normalize(data_df)
    # Add event_timestamp and patient_id columns to dataframe
    data_df = pd.concat(objs=[data_df, timestamps], axis=1)
    data_df = pd.concat(objs=[data_df, patient_ids], axis=1)
    label_df = pd.concat(objs=[label_df, timestamps], axis=1)
    label_df = pd.concat(objs=[label_df, patient_ids], axis=1)
    # Write dataframes to parquet files
    data_df.to_parquet('data.parquet')
    label_df.to_parquet('label.parquet')


def normalize(X):
    minX = X.min()
    range_X = (X - minX).max()
    X_scaled = (X - minX) / range_X
    return X_scaled


if __name__ == '__main__':
    print('Preparing data...')
    _data2parquet()
