# run_pipeline.py
import os.path
import pickle
import json
import sys
import argparse
import warnings

import pandas as pd
import numpy as np

from model import Model
from preprocessor import Preprocessor


class Pipeline:
    """
        A class representing a pipeline for data preprocessing and model training/testing.

        This pipeline consists of methods to initialize, fit, transform, and save preprocessing steps,
        as well as loading and using a machine learning model.

        Attributes:
        - target_feature (str): The target feature to predict.
        - cols_to_drop (list): List of columns to drop during preprocessing.
        - positive_threshold (float): Threshold for classifying positive instances.
        - preprocessor_path (str): File path to save/load the preprocessor object.
        - model_path (str): File path to save/load the model object.
        - preprocessor: The preprocessor object for data preprocessing.
        - model: The machine learning model for prediction.

        Methods:
        - __init__: Constructor method for the Pipeline class.
        - run: Method to run the pipeline for training/testing.
        - _run_test: Method to run the pipeline for testing.
        - _run_train: Method to run the pipeline for training.
        - _load_preprocessor: Method to load the preprocessor object from file.
        - _load_model: Method to load the model object from file.
        - _preprocessor_init: Method to initialize the preprocessor object.
        - _model_init: Method to initialize the model object.
        - _save_preprocessing: Method to save the preprocessor object to file.
        - _save_model: Method to save the model object to file.
        - _save_prediction: Static method to save predictions to file.
    """
    def __init__(self, target_feature: str, cols_to_drop: list, positive_threshold: float, preprocessor_path: str,
                 model_path: str):
        """
               Constructor method for the Pipeline class.

               Parameters:
               - target_feature (str): The target feature to predict.
               - cols_to_drop (list): List of columns to drop during preprocessing.
               - positive_threshold (float): Threshold for classifying positive instances.
               - preprocessor_path (str): File path to save/load the preprocessor object.
               - model_path (str): File path to save/load the model object.

               Returns:
               - None
        """

        self.target_feature = target_feature
        self.cols_to_drop = cols_to_drop
        self.positive_threshold = positive_threshold

        self.preprocessor_path = preprocessor_path
        self.model_path = model_path

        self.preprocessor = None
        self.model = None

    def run(self, X, test=False, preprocessor_params: dict = None, model_params: dict = None):
        """
                Method to run the pipeline for training/testing.

                Parameters:
                - X (DataFrame): Input data.
                - test (bool): Whether to run the pipeline for testing.
                - preprocessor_params (dict): Parameters for preprocessor initialization.
                - model_params (dict): Parameters for model initialization.

                Returns:
                - None
        """

        if test:
            self.preprocessor = self.__load_preprocessor()
            self.model = self.__load_model()
            self.__run_test(X)
        else:
            self.__preprocessor_init(preprocessor_params)
            self.__model_init(model_params)
            self.__run_train(X)
            self.__save_preprocessing()
            self.__save_model()

    def __run_test(self, X: pd.DataFrame):
        """
                Method to run the pipeline for testing.

                Parameters:
                - X (DataFrame): Input data for testing.

                Returns:
                - None
        """

        for col in self.cols_to_drop + [self.target_feature]:
            if col in X.columns:
                X = X.drop(col, axis=1)

        X = self.preprocessor.transform(X)
        predict_probas = self.model.predict(X)

        self.__save_prediction(predict_probas, self.positive_threshold)

    def __run_train(self, X: pd.DataFrame):
        """
                Method to run the pipeline for training.
                Parameters:
        - X (DataFrame): Input data for training.

        Returns:
        - None
        """

        for col in self.cols_to_drop:
            if col in X.columns:
                X = X.drop(col, axis=1)

        X, y = X.drop(columns=self.target_feature), X[self.target_feature]
        self.preprocessor.fit(X)
        X = self.preprocessor.transform(X)
        self.model.fit(X, y)

    def __load_preprocessor(self):
        """
                Method to load the preprocessor object from file.

                Returns:
                - preprocessor: Preprocessor object.
        """

        try:
            with open(self.preprocessor_path, "rb") as infile:
                preprocessor = pickle.load(infile)
                return preprocessor
        except FileNotFoundError:
            print("Fatal: Could not load preprocessor. Please specify correct path or run the pipeline on Train mode "
                  "to initialize the preprocessor")
            sys.exit()

    def __load_model(self):
        """
                Method to load the model object from file.

                Returns:
                - model: Model object.
        """

        try:
            with open(self.model_path, "rb") as infile:
                model = pickle.load(infile)
                return model
        except FileNotFoundError:
            print("Fatal: Could not load model. Please specify correct path or run the pipeline on Train mode to "
                  "initialize the model")
            sys.exit()

    def __preprocessor_init(self, preprocessor_params: dict):
        """
               Method to initialize the preprocessor object.

               Parameters:
               - preprocessor_params (dict): Parameters for preprocessor initialization.

               Returns:
               - None
        """

        nan_drop_threshold = 0.5
        imputing_strategy = "median"
        scaling_strategy = "StandardScaler"
        if preprocessor_params is not None:
            nan_drop_threshold = preprocessor_params.get("nan_drop_threshold", 0.5)
            imputing_strategy = preprocessor_params.get("imputing_strategy", "median")
            scaling_strategy = preprocessor_params.get("scaling_strategy", "StandardScaler")

        self.preprocessor = Preprocessor(nan_drop_threshold, imputing_strategy, scaling_strategy)

    def __model_init(self, model_params: dict):
        """
                Method to initialize the model object.

                Parameters:
                - model_params (dict): Parameters for model initialization.

                Returns:
                - None
        """

        n_estimators = 500
        max_depth = None
        min_samples_split = 2
        sampling_strategy = "all"
        if model_params is not None:
            n_estimators = model_params.get("n_estimators", 500)
            max_depth = model_params.get("max_depth", None)
            min_samples_split = model_params.get("min_samples_split", 2)
            sampling_strategy = model_params.get("sampling_strategy", "all")

        self.model = Model(n_estimators, max_depth, min_samples_split, sampling_strategy)

    def __save_preprocessing(self):
        """
                Method to save the preprocessor object to file.

                Returns:
                - None
        """

        if not os.path.exists(os.path.dirname(self.preprocessor_path)):
            os.makedirs(os.path.dirname(self.preprocessor_path))

        with open(self.preprocessor_path, "wb") as outfile:
            pickle.dump(self.preprocessor, outfile)

    def __save_model(self):
        """
                Method to save the model object to file.

                Returns:
                - None
        """

        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))

        with open(self.model_path, "wb") as outfile:
            pickle.dump(self.model, outfile)

    @staticmethod
    def __save_prediction(predict_probas: np.ndarray, threshold: float):
        """
               Static method to save predictions to file.

               Parameters:
               - predict_probas (np.ndarray): Predicted probabilities.
               - threshold (float): Classification threshold.

               Returns:
               - None
        """

        predict_probas = list(predict_probas)
        result = {"predict_probas": predict_probas, "threshold": threshold}
        json_object = json.dumps(result)

        # Writing to sample.json
        with open("predictions.json", "w+") as outfile:
            outfile.write(json_object)


# run code
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--test", action='store_true')
parser.add_argument("--data_path", type=str)
parser.add_argument("--preprocessor_path", type=str, default="preprocessor.sav")
parser.add_argument("--model_path", type=str, default="model.sav")
parser.add_argument("--threshold", type=float, default=0.45)

args = parser.parse_args()

PREPROCESSOR_PARAMS = {"nan_drop_threshold": 0.5, "imputing_strategy": "median", "scaling_strategy": "StandardScaler"}
MODEL_PARAMS = {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "sampling_strategy": "all"}

df = pd.read_csv(args.data_path)
pipeline = Pipeline("In-hospital_death", ["recordid", "SAPS-I", "SOFA", "Length_of_stay", "Survival"],
                    args.threshold, args.preprocessor_path, args.model_path)
pipeline.run(df, test=args.test, preprocessor_params=PREPROCESSOR_PARAMS, model_params=MODEL_PARAMS)
