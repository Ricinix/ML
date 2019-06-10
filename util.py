import numpy as np


class DataPreHandle:

    @staticmethod
    def min_max_normalization(X):
        for n in range(X.shape[1]):
            X[:, n] = (X[:, n] - np.min(X[:, n])) / (np.max(X[:, n]) - np.min(X[:, n]))
        return X

    # z-score 标准化(正态分布标准化)
    @staticmethod
    def zero_mean_normalization(X):
        for n in range(X.shape[1]):
            X[:, n] = (X[:, n] - X[:, n].mean()) / X[:, n].std()
        return X