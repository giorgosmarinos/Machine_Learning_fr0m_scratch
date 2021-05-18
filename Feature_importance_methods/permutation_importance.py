import pandas as pd
import sklearn.metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import mean_squared_error

class permutation_importance():

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def train_with_non_permuted_features(self):

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        r2_score = sklearn.metrics.r2_score(self.y_test, preds)
        m2error = mean_squared_error(self.y_test, preds)
        return r2_score, m2error

    def permute_features(self):

        lista = []

        for name, values in self.X_train.iteritems():
            self.X_train[name] = np.random.permutation(self.X_train[name].values)

            #train the algorithm
            clf_per = DecisionTreeClassifier(random_state=42)
            clf_per.fit(self.X_train, self.y_train)
            preds = clf_per.predict(self.X_test)
            r2_score = sklearn.metrics.r2_score(self.y_test, preds )
            m2error = mean_squared_error(self.y_test, preds)
            lista.append([name,r2_score,m2error])

        return lista


if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    perm = permutation_importance(X_train, y_train, X_test, y_test)

    r2_score_orig, m2error_orig = perm.train_with_non_permuted_features()

    lista = perm.permute_features()

    for i in lista:
        print(m2error_orig-i[2])

