from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

#data = load_boston()

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)