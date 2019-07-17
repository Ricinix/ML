import os
import numpy as np
import pandas as pd
from util import DataPreHandle


path = os.path.join('data')
name = 'train_level.pickle'


# sigmoid函数
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


# l函数
def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# logistics回归主体
def logistic_reg(
    alpha,
    X,
    y,
    max_iterations=70000
    ):
    converged = False
    iterations = 0
    theta = np.zeros(X.shape[1])

    while not converged:
        z = np.dot(X, theta)
        h = sigmoid_function(z)
        gradient = np.dot(X.T, h - y) / y.size  # 普通梯度下降，甚至省略了迭代
        theta = theta - alpha * gradient

        z = np.dot(X, theta)
        h = sigmoid_function(z)
        J = cost_function(h, y)

        print(iterations)
        iterations += 1

        if iterations == max_iterations:
            print('Maximum iterations exceeded!')
            print('Minimal cost function J=', J)
            converged = True

    return theta


if __name__ == "__main__":

    # 加载数据
    data_train = pd.read_pickle(os.path.join(path, name))
    np_data = np.array(data_train)
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np_data[:, 0:n]
    y = np_data[:, n]
    # 数据归一化
    X = DataPreHandle.zero_mean_normalization(X)
    alpha = 0.1
    theta = logistic_reg(alpha, X, y, max_iterations=100000)
    print(theta)
    df = pd.DataFrame(theta)
    df.to_pickle(os.path.join('.', 'module', "logistic_reg_module.pickle"))
