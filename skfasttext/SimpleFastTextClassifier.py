from sklearn.base import BaseEstimator, ClassifierMixin
import fastText as ft
from sklearn.metrics import classification_report

class SimpleFastTextClassifier(BaseEstimator,ClassifierMixin):
    """
    Base classiifer of FastText estimator
    """

    def _init__(self):
        self.classifier=None
        self.result=None

    def fit(self,train_file):
            self.classifier = ft.train_supervised(train_file)
            return(self)

    def predict(self, test_file, k_best=1):
        predictions = []
        with open(test_file) as f:
            for line in f:
                predictions.append(self.classifier.predict(line.strip(), k=k_best))
        self.result=predictions
        return self.result
