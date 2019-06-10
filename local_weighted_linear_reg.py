import os
import time
import numpy as np
import pandas as pd
from util import DataPreHandle


def load_data(name):
    return pd.read_pickle(os.path.join('.', 'data', name))


def get_weights(X, x_test, k=1.0):
    diff = X - x_test
    temp = diff.dot(diff.T)
    temp = temp * np.eye(diff.shape[0])
    temp = np.array(temp[temp > 0])
    w = np.exp(-temp / (2 * k**2))
    return w


def get_weights_iteration(X, x_test, k=1.0):
    diff = X - x_test
    temp = np.zeros(diff.shape[0])

    for i in range(temp.shape[0]):
        temp[i] = diff[i].dot(diff[i])

    w = np.exp(-temp / (2 * k**2))
    return w


def get_weights_time_test(X, x_test):
    start = time.time()
    get_weights(X, x_test)
    end = time.time()
    no_iterations_time = end - start

    start = time.time()
    get_weights_iteration(X, x_test)
    end = time.time()
    iterations_time = end - start
    print("无迭代方法所耗时间：%.5f\n迭代方法所耗时间：%.5f" % (no_iterations_time, iterations_time))


def run_steep_gradient_descent(X, y, alpha, theta, weights):
    prod = (np.dot(X, theta) - y) * weights
    sum_grad = np.dot(prod, X)
    theta = theta - (alpha / X.shape[0]) * sum_grad
    return theta


def sum_of_square_error(X, y, theta, weights):
    prod = (np.dot(X, theta) - y) * weights
    error = (np.square(prod)).mean()
    return error


def local_weighted_linear_reg(X, y, x_test, iterations = 70000, alpha = 0.001):
    theta = np.zeros(X.shape[1])
    weights = get_weights_iteration(X, x_test, 1.0)

    for i in range(iterations):
        theta = run_steep_gradient_descent(X, y, alpha, theta, weights)
        error = sum_of_square_error(X, y, theta, weights)
        print('At Iteration %d - Error is %.5f ' % (i + 1, error))

    return theta


def mse(result, y_t):
    mse = np.square(result - y_t)
    return mse.mean()


def R_squared(result, y_t):
    residual = np.square(result - y_t)
    residual = residual.sum()

    total = np.square(y_t.mean() - y_t)
    total = total.sum()

    r_square = 1 - residual / total
    return r_square


if __name__ == '__main__':
    data_train = load_data('train_shares.pickle')
    np_data = np.array(data_train)
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np_data[:, 0:n]
    y = np_data[:, n]
    X = DataPreHandle.min_max_normalization(X)

    data_verify = load_data('verification_shares.pickle')
    np_data_verify = np.array(data_verify)
    m_t = data_verify.shape[0]
    X_t = np_data_verify[:, 0:n]
    y_t = np_data_verify[:, n]
    X_t = DataPreHandle.min_max_normalization(X_t)

    # get_weights_time_test(X, X_t[0])
    result = np.zeros(m_t)
    for i in range(m_t):
        theta = local_weighted_linear_reg(X, y, X_t[i])
        result[i] = X_t[i].dot(theta)

    r_square = R_squared(result, y_t)
    print('R-squared:', r_square)
    mse = mse(result, y)
    print('MSE:', mse)
