from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def haversine_vectorized(self, df):
        """
            Calculates the great circle distance between two points
            on the earth (specified in decimal degrees).
            Vectorized version of the haversine distance for pandas df.
            Computes the distance in kms.
        """
        #use self to access attributes
        lat_1_rad, lon_1_rad = np.radians(
            df[self.start_lat].astype(float)), np.radians(
                df[self.start_lon].astype(float))
        lat_2_rad, lon_2_rad = np.radians(
            df[self.end_lat].astype(float)), np.radians(df[self.end_lon].astype(float))
        dlon = lon_2_rad - lon_1_rad
        dlat = lat_2_rad - lat_1_rad

        a = np.sin(
            dlat / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(
                dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = self.haversine_vectorized(X_)
        return X_[['distance']]


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]
