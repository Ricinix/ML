import pandas as pd
import numpy as np


path = ".\\data\\"
data_name = 'verification_shares.pickle'
module_name = 'linear_reg_moudle.pickle'
data_train = pd.read_pickle(path + data_name)
theta = pd.read_pickle(path + module_name)


# min-max标准化(线性标准化)
def min_max_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - np.max(X[m])) / (np.max(X[m]) - np.min(X[m]))
    return X


def zero_mean_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - X[m].mean()) / X[m].std()
    return X


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
    X = min_max_normalization(X)

    r_square = R_squared(X, y)
    print('R-squared:', r_square)
    mse = mse(X, y)
    print('MSE:', mse)