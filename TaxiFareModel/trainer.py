import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from data import get_data, clean_data, set_features_targets, holdout
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import math
import joblib


class Trainer():
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        # self.X = X
        # self.y = y

    def create_pipeline(self, model_name, model):
        dist_pipe = Pipeline([
            ('dis_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])

        #time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        # training pipeline
        self.pipe = Pipeline([('preproc', preproc_pipe), (model_name, model)])

    def evaluate_pipe(self, X_train, X_test, y_train, y_test, model_name):
        # train the pipelined model
        self.pipe.fit(X_train, y_train)
        #score = self.pipe.score(X_test, y_test)
        y_pred = self.pipe.predict(x_test)
        score = np.sqrt(((y_pred - y_test)**2).mean())
        print(f"score with {model_name}", score)


    def train(self, x_train, x_test, y_train, y_test, model_name, model):
        self.create_pipeline(model_name, model)
        self.evaluate_pipe(x_train, x_test, y_train, y_test, model_name)

    def save_model(self):
        joblib.dump(self.pipe, 'model.joblib')



if __name__ == "__main__":
    #trainer = Trainer()
    df = get_data()
    df = clean_data(df)
    x, y = set_features_targets(df)
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2)

    models = [
        ('linreg', LinearRegression()),
        ('KNN', KNeighborsRegressor(n_neighbors=10)),
        ('XGB', XGBRegressor())]

    trainer = Trainer()
    for model_name, model in models:
        trainer.train(x_train, x_test, y_train, y_test, model_name, model)
        #print()
