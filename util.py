import numpy as np
import pandas as pd


class DataPreHandle:

    # min-max标准化(线性标准化)
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

    # 把连续值分成10段的离散值
    @staticmethod
    def discrete_normalization(data, *args):
        columns = data.columns[0: -1]
        for column in columns:
            if column in args:
                continue
            temp = (data[column].max() - data[column].min()) / 10
            min = data[column].min()
            data[column] = (data[column] >= 9 * temp + min) * 1 +\
                (data[column] >= 8 * temp + min) * 1 +\
                (data[column] >= 7 * temp + min) * 1 +\
                (data[column] >= 6 * temp + min) * 1 +\
                (data[column] >= 5 * temp + min) * 1 +\
                (data[column] >= 4 * temp + min) * 1 +\
                (data[column] >= 3 * temp + min) * 1 +\
                (data[column] >= 2 * temp + min) * 1 +\
                (data[column] >= 1 * temp + min) * 1
        return data
