import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv('../raw_data/train.csv', nrows = nrows)
    return df

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def set_features_targets(df):
    y = df.pop('fare_amount')
    x = df
    return x, y


def holdout(x, y, test_size=0.3):
    """
    Splits data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test
