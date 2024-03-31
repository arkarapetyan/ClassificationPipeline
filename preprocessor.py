# preprocessor.py
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    """
        A class representing a data preprocessor for dropping columns with much nan values, handling missing values and scaling features.

        Attributes:
        - nan_drop_threshold (float): Threshold for dropping columns with a high percentage of missing values.
        - imputer: The imputer object for handling missing values.
        - scaler: The scaler object for scaling features.
        - cols_to_drop (list): List of column names to be dropped due to high percentage of missing values.

        Methods:
        - __init__: Constructor method for the Preprocessor class.
        - fit: Method to fit the preprocessor on the training data.
        - transform: Method to transform the input data using the fitted preprocessor.
        - _get_imputer_by_name: Static method to get the imputer object based on the specified name.
        - _get_scaler_by_name: Static method to get the scaler object based on the specified name.
    """
    def __init__(self, nan_drop_threshold: float, imputing_strategy: str, scaling_strategy: str):
        """
               Constructor method for the Preprocessor class.

               Parameters:
               - nan_drop_threshold (float): Threshold for dropping columns with a high percentage of missing values.
               - imputing_strategy (str): Strategy used for imputing missing values.
               - scaling_strategy (str): Strategy used for scaling features.

               Returns:
               - None
        """

        self.nan_drop_threshold = nan_drop_threshold
        self.imputer = self.__get_imputer_by_name(imputing_strategy)
        self.scaler = self.__get_scaler_by_name(scaling_strategy)
        self.cols_to_drop = None

    def fit(self, X):
        """
                Method to fit the preprocessor on the training data.

                Parameters:
                - X (DataFrame): Training data.

                Returns:
                - None
        """

        count = X.isna().sum()
        self.cols_to_drop = list(X.columns[count / len(X) > self.nan_drop_threshold])
        X = X.drop(columns=self.cols_to_drop)

        self.imputer.fit(X)
        self.scaler.fit(X)

    def transform(self, X):
        """
                Method to transform the input data using the fitted preprocessor.

                Parameters:
                - X (DataFrame): Input data to be transformed.

                Returns:
                - X_transformed (DataFrame): Transformed data.
        """

        X = X.drop(columns=self.cols_to_drop)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        return X

    @staticmethod
    def __get_imputer_by_name(name: str):
        """
                Static method to get the imputer object based on the specified name.

                Parameters:
                - name (str): Name of the imputation strategy.

                Returns:
                - imputer: Imputer object.
        """

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
    def __get_scaler_by_name(name: str):
        """
               Static method to get the scaler object based on the specified name.

               Parameters:
               - name (str): Name of the scaling strategy.

               Returns:
               - scaler: Scaler object.
        """
        
        if name == "MinMaxScaler":
            return MinMaxScaler()
        elif name == "StandardScaler":
            return StandardScaler()
        else:
            return None
