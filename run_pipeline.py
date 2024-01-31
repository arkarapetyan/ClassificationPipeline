# run_pipeline.py

import pickle
import json
import sys
import argparse

import pandas as pd
import numpy as np

from model import Model
from preprocessor import Preprocessor


class Pipeline:
    def __init__(self, target_feature: str, cols_to_drop: list, positive_threshold: float, preprocessor_path: str,
                 model_path: str):

        self.target_feature = target_feature
        self.cols_to_drop = cols_to_drop
        self.positive_threshold = positive_threshold

        self.preprocessor_path = preprocessor_path
        self.model_path = model_path

        self.preprocessor = None
        self.model = None

    def run(self, X, test=False, preprocessor_params: dict = None, model_params: dict = None):
        if test:
            self.preprocessor = self._load_preprocessor()
            self.model = self._load_model()
            self._run_test(X)
        else:
            self._preprocessor_init(preprocessor_params)
            self._model_init(model_params)
            self._run_train(X)
            self._save_preprocessing()
            self._save_model()

    def _run_test(self, X: pd.DataFrame):
        for col in self.cols_to_drop + [self.target_feature]:
            if col in X.columns:
                X = X.drop(col, axis=1)

        X = self.preprocessor.transform(X)
        predict_probas = self.model.predict(X)

        self._save_prediction(predict_probas, self.positive_threshold)

    def _run_train(self, X: pd.DataFrame):
        for col in self.cols_to_drop:
            if col in X.columns:
                X = X.drop(col, axis=1)

        X, y = X.drop(columns=self.target_feature), X[self.target_feature]
        self.preprocessor.fit(X)
        X = self.preprocessor.transform(X)
        self.model.fit(X, y)

    def _load_preprocessor(self):
        try:
            with open(self.preprocessor_path, "rb") as infile:
                preprocessor = pickle.load(infile)
                return preprocessor
        except FileNotFoundError:
            print("Fatal: Could not load preprocessor. Please specify correct path or run the pipeline on Train mode "
                  "to initialize the preprocessor")
            sys.exit()
            return None

    def _load_model(self):
        try:
            with open(self.model_path, "rb") as infile:
                model = pickle.load(infile)
                return model
        except FileNotFoundError:
            print("Fatal: Could not load model. Please specify correct path or run the pipeline on Train mode to "
                  "initialize the model")
            sys.exit()
            return None

    def _preprocessor_init(self, preprocessor_params: dict):
        nan_drop_threshold = 0.5
        imputing_strategy = "median"
        scaling_strategy = "StandardScaler"
        if preprocessor_params is not None:
            nan_drop_threshold = preprocessor_params.get("nan_drop_threshold", 0.5)
            imputing_strategy = preprocessor_params.get("imputing_strategy", "median")
            scaling_strategy = preprocessor_params.get("scaling_strategy", "StandardScaler")

        self.preprocessor = Preprocessor(nan_drop_threshold, imputing_strategy, scaling_strategy)

    def _model_init(self, model_params: dict):
        # TODO
        self.model = Model()

    def _save_preprocessing(self):
        with open(self.preprocessor_path, "wb") as outfile:
            pickle.dump(self.preprocessor, outfile)

    def _save_model(self):
        with open(self.model_path, "wb") as outfile:
            pickle.dump(self.model, outfile)

    @staticmethod
    def _save_prediction(predict_probas: np.ndarray, threshold: float):
        predict_probas = list(predict_probas)
        result = {"predict_probas": predict_probas, "threshold": threshold}
        json_object = json.dumps(result)

        # Writing to sample.json
        with open("predictions.json", "w") as outfile:
            outfile.write(json_object)


parser = argparse.ArgumentParser()
parser.add_argument("--test", type=bool, default=False)
parser.add_argument("--data_path", type=str)
parser.add_argument("--preprocessor_path", type=str, default="preprocessor.sav")
parser.add_argument("--model_path", type=str, default="model.sav")

args = parser.parse_args()

preprocessor_params = {"nan_drop_threshold": 0.5, "imputing_strategy": "median", "scaling_strategy": "StandardScaler"}
model_params = {}
df = pd.read_csv(args.data_path)

pipeline = Pipeline("In-hospital_death", ["recordid", "SAPS-I", "SOFA", "Length_of_stay", "Survival"],
                    0.5, args.preprocessor_path, args.model_path)
pipeline.run(df, test=args.test, preprocessor_params=preprocessor_params, model_params=model_params)
