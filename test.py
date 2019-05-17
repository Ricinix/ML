from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, 2]
y = (iris.target != 0) * 1
print(X)
print(y)