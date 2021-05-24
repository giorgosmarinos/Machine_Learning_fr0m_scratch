import pandas as pd
import sklearn.metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge


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
        times = 10
        importances = {}

        for name, values in self.X_train.iteritems():

            scores = []

            for k in range(1, times+1):

                self.X_train[name] = np.random.permutation(self.X_train[name].values)

                #train the algorithm
                clf_per = DecisionTreeClassifier(random_state=42)
                clf_per.fit(self.X_train, self.y_train)
                preds = clf_per.predict(self.X_test)

                #compute the scores R2_score, m2_error, importance
                r2_score = sklearn.metrics.r2_score(self.y_test, preds)
                m2error = mean_squared_error(self.y_test, preds)
                scores.append(m2error)

                lista.append([name,r2_score,m2error, scores])

            importance = m2error - (1 / (times + 1)) * sum(scores) #TODO implement the function https://scikit-learn.org/stable/modules/permutation_importance.html
            importances.update({name:importance})

        return lista, importances


if __name__ == "__main__":

    cancer = load_breast_cancer()

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    X_train = pd.DataFrame(X_train, columns=cancer['feature_names'])
    X_test = pd.DataFrame(X_test, columns=cancer['feature_names'])
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    """print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)"""

    perm = permutation_importance(X_train, y_train, X_test, y_test)

    r2_score_orig, m2error_orig = perm.train_with_non_permuted_features()

    lista, importancess = perm.permute_features()

    for i in lista:
        print(r2_score_orig-i[2])

    print("#######################################################################")
    print("\n")

    print("My finding are ...")
    print("\n")
    print(pd.DataFrame.from_dict(importancess.items()))


    print("######################################################################")
    print("In Comparison with .... ")
    print("\n")

    breast_cancer = load_breast_cancer()
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

    model = Ridge(alpha=1e-2).fit(X_train, y_train)
    model.score(X_val, y_val)

    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X_val, y_val,
                               n_repeats=30,
                               random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{breast_cancer.feature_names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

