import os
import numpy as np
import pandas as pd
from util import DataPreHandle


def run_steep_gradient_descent(X, y, alpha, theta):
    prod = np.dot(X, theta) - y
    sum_grad = np.dot(prod, X)
    theta = theta - (alpha / X.shape[0]) * sum_grad
    return theta


def sum_of_square_error(X, y, theta):
    prod = np.dot(X, theta) - y
    error = (1 / 2) * (np.square(prod)).mean()
    return error


def run_linear_regression(X, y):
    iterations = 100000
    alpha = 0.2

    theta = np.zeros(X.shape[1])

    for i in range(iterations):
        theta = run_steep_gradient_descent(X, y, alpha, theta)
        error = sum_of_square_error(X, y, theta)
        print('At Iteration %d - Error is %.5f ' % (i + 1, error))

    return theta


if __name__ == '__main__':
    data_train = pd.read_pickle(os.path.join('.', 'data', 'train_shares.pickle'))
    np_data = np.array(data_train)
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np_data[:, 0:n]
    y = np_data[:, n]
    X = DataPreHandle.min_max_normalization(X)

    theta = run_linear_regression(X, y)
    print(theta)
    df = pd.DataFrame(theta)
    df.to_pickle(os.path.join('.', 'module', 'linear_reg_module.pickle'))