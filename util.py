import numpy as np


class data_pre_handle:

    @staticmethod
    def min_max_normalization(X):
        for m in range(X.shape[0]):
            X[m] = (X[m] - np.max(X[m])) / (np.max(X[m]) - np.min(X[m]))
        return X

    # z-score 标准化(正态分布标准化)
    @staticmethod
    def zero_mean_normalization(X):
        for m in range(X.shape[0]):
            X[m] = (X[m] - X[m].mean()) / X[m].std()
        return X