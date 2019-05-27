import numpy as np
import pandas as pd


path = "E:\\Programming\\python_pickle\\"
name = 'train_shares.pickle'


# min-max标准化(线性标准化)
def min_max_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - np.max(X[m])) / (np.max(X[m]) - np.min(X[m]))
    return X


# z-score 标准化(正态分布标准化)
def zero_mean_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - X[m].mean()) / X[m].std()
    return X


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
    alpha = 0.02

    theta = np.zeros(X.shape[1])

    for i in range(iterations):
        theta = run_steep_gradient_descent(X, y, alpha, theta)
        error = sum_of_square_error(X, y, theta)
        print('At Iteration %d - Error is %.5f ' % (i + 1, error))

    return theta


if __name__ == '__main__':
    data_train = pd.read_pickle(path + name)
    np_data = np.array(data_train)
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np_data[:, 0:n]
    y = np_data[:, n]
    X = min_max_normalization(X)

    theta = run_linear_regression(X, y)
    print(theta)
    df = pd.DataFrame(theta)
    df.to_pickle(path + 'linear_reg_moudle.pickle')