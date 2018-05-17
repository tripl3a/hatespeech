from sklearn.base import BaseEstimator, ClassifierMixin
import fastText as ft
from sklearn.metrics import classification_report
import pandas as pd
import csv
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class SimpleFastTextClassifier(BaseEstimator,ClassifierMixin):
    """
    Base classiifer of FastText estimator
    """

    #def _init__(self):
        #self.classifier=None
        #self.results=None

    def fit(self, X, y):
        # Check that X and y have correct shape
        #X, y = check_X_y(X.values.reshape(-1, 1), y, dtype="str")
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)
        df = pd.concat([df_X, df_y], axis=1)
        f = "data/train_data.txt"
        df.to_csv(f, index=False, sep=" ", header=False, escapechar=" ", quoting=csv.QUOTE_NONE)
        
        self.X_ = X
        self.y_ = y
        self.classifier = ft.train_supervised(f)
        
        return(self)
    
    def score(self, X, y):
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)
        df = pd.concat([df_X, df_y], axis=1)
        
        f = "data/test_data.txt"
        df.to_csv(f, index=False, sep=" ", header=False, escapechar=" ", quoting=csv.QUOTE_NONE)
        
        results = self.classifier.test(f)
        return results[1] #precision
    
    def print_results(self, N, p, r):
        #N, p, r = self.results
        # see https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/train_supervised.py
            print("N\t" + str(N))
            print("P@{}\t{:.3f}".format(1, p)) #precision
            print("R@{}\t{:.3f}".format(1, r)) #recall
    
    def pre_predict(self, X, k_best=1):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X.values.reshape(-1, 1), dtype="str")
        
        df = pd.DataFrame(X)
        f = "data/test_data.txt"
        df.to_csv(f, index=False, sep=" ", header=False, escapechar=" ", quoting=csv.QUOTE_NONE)
        return f

    def predict(self, X, k_best=1):
        test_file = self.pre_predict(X, k_best)
        preds = []
        with open(test_file) as f:
            for line in f:
                preds.append(self.classifier.predict(line.strip(), k=k_best)[0][0])
        return preds
    
    def predict_proba(self, X, k_best=1):
        test_file = self.pre_predict(X, k_best)
        preds_proba = []
        with open(test_file) as f:
            for line in f:
                preds_proba.append(self.classifier.predict(line.strip(), k=k_best)[1][0])
        return preds_proba
