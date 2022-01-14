from sklearn.tree import DecisionTreeRegressor
import pandas as pd 
from csv import reader
import numpy as np 
from sklearn.metrics import mean_squared_error

class GradientBoostingFromScratch():
    
    def __init__(self, n_trees, learning_rate, max_depth=1):
        self.n_trees=n_trees 
        self.learning_rate=learning_rate 
        self.max_depth=max_depth
        
    def fit(self, x, y):
        self.trees = []
        self.F0 = y.mean()
        Fm = self.F0 
        for _ in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(x, y - Fm)
            Fm += self.learning_rate * tree.predict(x)
            self.trees.append(tree)
            
    def predict(self, x):
        return self.F0 + self.learning_rate * np.sum([tree.predict(x) for tree in self.trees], axis=0)


# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# load and prepare data
filename = 'D:\\PycharmProjects\\Machine_Learning_from_scratch\\Datasets\\classification\\data_banknote_authentication.txt'
dataset = load_csv(filename)


# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)

dataset = pd.DataFrame(dataset, columns=['X_1', 'X_2', 'X_3', 'X_4', 'X_5'])

print(dataset)

x = dataset[['X_1', 'X_2', 'X_3', 'X_4']]
y = dataset['X_5']

scratch_gbm = GradientBoostingFromScratch(n_trees=25, learning_rate=0.3, max_depth=1)
scratch_gbm.fit(x,y)

print(mean_squared_error(y, scratch_gbm.predict(x)))
