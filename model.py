from imblearn.ensemble import BalancedRandomForestClassifier


class Model:
    """
       A class representing a machine learning model based on Balanced Random Forest Classifier.

       Attributes:
       - clf (BalancedRandomForestClassifier): The Balanced Random Forest Classifier instance.

       Methods:
       - __init__: Constructor method for the Model class.
       - fit: Method to fit the model to the training data.
       - predict: Method to make predictions on new data.
       """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2,
                 sampling_strategy: str = "all"):
        """
                Constructor method for the Model class.

                Parameters:
                - n_estimators (int): Number of trees in the forest.
                - max_depth (int): Maximum depth of the tree.
                - min_samples_split (int): Minimum number of samples required to split an internal node.
                - sampling_strategy (str): Strategy used to resample the dataset.

                Returns:
                - None
                """

        self.clf = BalancedRandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  sampling_strategy=sampling_strategy)

    def fit(self, X, y):
        """
                Method to fit the model to the training data.

                Parameters:
                - X (array-like, shape (n_samples, n_features)): Training data.
                - y (array-like, shape (n_samples,)): Target values.

                Returns:
                - None
                """
        self.clf.fit(X, y)

    def predict(self, X):
        """
                Method to make predictions on new data.

                Parameters:
                - X (array-like, shape (n_samples, n_features)): New data to make predictions on.

                Returns:
                - y_proba (array-like, shape (n_samples,)): Predicted probabilities for the positive class.
                """
        y_proba = self.clf.predict_proba(X)[:, 1]
        return y_proba