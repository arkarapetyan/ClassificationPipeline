from imblearn.ensemble import BalancedRandomForestClassifier


class Model:
    def __init__(self, n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2,
                 sampling_strategy: str = "all"):
        self.clf = BalancedRandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  sampling_strategy=sampling_strategy)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        y_proba = self.clf.predict_proba(X)[:, 1]
        return y_proba
