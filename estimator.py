import os
import joblib
import datetime
import traceback

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Estimator:
    def __init__(self, model_path: str = None, inference_mode: bool = False):
        '''
            According to EDA and series of experiments,
            simple Linear regression is a suitable estimator
            for this task.
            Clear linear relation between target and two features was found.
            Following the Occam's razor, and AIC and BIC criteria it is better to choose the simpliest estimator.
            Also, it is the smallest one, comparing by serialized files size (just two parameters).
            Model retrain would help to adjust the model two some minor changes in features distributions.
            But in case of linear dependency violation the whole investigation should be started from the beggining.
        '''
        self._current_features_in_use = ['6', '7']
        self._final_feature_list = ['6_squared', '7']
        self._target_col = 'target'
        self.model_path = os.getcwd() if model_path is None else model_path
        if inference_mode:
            try:
                self._regressor = joblib.load(self.model_path)
            except Exception as e:
                print('Cannot load model')
                traceback.print_exc()
        else:
            self._regressor = LinearRegression()


    @staticmethod
    def root_mean_squared_error(y_test, y_pred) -> float:
        return np.sqrt(mean_squared_error(y_test, y_pred))

    def validate(self, X_test, y_test) -> float:
        y_pred = self._regressor.predict(X_test)
        rmse = Estimator.root_mean_squared_error(y_test, y_pred)
        print(f'Val RMSE: {rmse}')
        return rmse

    @staticmethod
    def _load_dataset(dataset: pd.DataFrame = None, dataset_path: str = None) -> pd.DataFrame:
        if dataset is None and dataset_path is None:
            raise Exception('Either dataset or dataset_path should be provided! ')
        elif dataset is None:
            if not os.path.exists(dataset_path):
                raise Exception('Dataset path doest not exist!')
            df = pd.read_csv(dataset_path)
        else:
            df = dataset
        return df

    def fit(self, dataset: pd.DataFrame = None, dataset_path: str = None, test_size: float = 0.1) -> None:

        df = Estimator._load_dataset(dataset=dataset, dataset_path=dataset_path)

        # Feature engineering
        df = df[[*self._current_features_in_use, 'target']]
        df['6_squared'] = df['6'] ** 2
        df.drop(['6'], axis=1, inplace=True)

        # Splitting dataset for training and validation
        X = df[self._final_feature_list]
        y = df[[self._target_col]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Fitting regressor
        self._regressor.fit(X_train, y_train)

        # Validate
        self.validate(X_test, y_test)

    def save_model(self) -> None:
        model_name = f'estimator_{datetime.datetime.now().strftime(format='%Y-%m-%d_%H-%M-%S')}.joblib'
        joblib.dump(self._regressor, os.path.join(self.model_path, model_name))

    def predict(self, features_dataset: pd.DataFrame = None, feature_dataset_path: str = None,
                save_preds_to_csv=False) -> np.array:
        df = Estimator._load_dataset(dataset=features_dataset, dataset_path=feature_dataset_path)

        # Feature engineering
        df_inference = df[self._current_features_in_use].copy() # may use df object later to write it with prediction
        df_inference.loc[:, '6_squared'] = df_inference['6'] ** 2
        df_inference.drop(['6'], axis=1, inplace=True)
        X = df_inference[self._final_feature_list]

        # Predict
        y_pred = self._regressor.predict(X)

        # Save
        if save_preds_to_csv:
            df['prediction'] = y_pred
            dir_name, file_name = os.path.split(feature_dataset_path)
            new_file_name = f'{file_name.split(".")[0]}_with_predictions.csv'
            df.to_csv(os.path.join(dir_name, new_file_name))
        return y_pred
