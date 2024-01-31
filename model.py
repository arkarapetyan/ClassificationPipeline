from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self,):
        self.clf = RandomForestClassifier()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        y_proba = self.clf.predict_proba(X)[:, 1]
        return y_proba
