import os
import pandas as pd
import numpy as np
from util import DataPreHandle


data_train = pd.read_pickle(os.path.join('.', 'data', 'verification_shares.pickle'))
theta = pd.read_pickle(os.path.join('.', 'module', 'linear_reg_moudle.pickle'))


def mse(X, y):
    mse = np.square(np.dot(X, theta) - y)
    return mse.mean()


def R_squared(X, y):
    residual = np.square(np.dot(X, theta) - y)
    residual = residual.sum()

    total = np.square(y.mean() - y)
    total = total.sum()

    r_square = 1 - residual / total
    return r_square


if __name__ == '__main__':
    # 加载数据
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np.array(data_train)
    y = X[:, n]
    X = X[:, :n]
    # 数据归一化
    X = DataPreHandle.min_max_normalization(X)

    r_square = R_squared(X, y)
    print('R-squared:', r_square)
    mse = mse(X, y)
    print('MSE:', mse)
