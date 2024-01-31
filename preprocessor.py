# preprocessor.py
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    def __init__(self, nan_drop_threshold: float, imputing_strategy: str, scaling_strategy: str):
        self.nan_drop_threshold = nan_drop_threshold
        self.imputer = self._get_imputer_by_name(imputing_strategy)
        self.scaler = self._get_scaler_by_name(scaling_strategy)
        self.cols_to_drop = None

    def fit(self, X):
        count = X.isna().sum()
        self.cols_to_drop = list(X.columns[count / len(X) > self.nan_drop_threshold])
        X = X.drop(columns=self.cols_to_drop)

        self.imputer.fit(X)
        self.scaler.fit(X)

    def transform(self, X):
        X = X.drop(columns=self.cols_to_drop)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        return X

    @staticmethod
    def _get_imputer_by_name(name: str):
        if name == "median":
            return SimpleImputer(strategy="median")
        elif name == "mean":
            return SimpleImputer(strategy="mean")
        elif name == "most-frequent":
            return SimpleImputer(strategy="most-frequent")
        elif name == "knn":
            return KNNImputer(n_neighbors=3, weights="distance")
        else:
            return None

    @staticmethod
    def _get_scaler_by_name(name: str):
        if name == "MinMaxScaler":
            return MinMaxScaler()
        elif name == "StandardScaler":
            return StandardScaler()
        else:
            return None


