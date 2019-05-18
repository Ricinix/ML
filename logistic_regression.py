import numpy as np
import pandas as pd


path = "E:\\Programming\\python_pickle\\"
name = 'train_level.pickle'


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
    data_train = pd.read_pickle(path + name)
    np_data = np.array(data_train)
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np_data[:, 0:n]
    y = np_data[:, n]
    # 数据归一化
    X = zero_mean_normalization(X)

    alpha = 0.1
    theta = logistic_reg(alpha, X, y, max_iterations=100000)
    print(theta)
    df = pd.DataFrame(theta)
    df.to_pickle(path + "logistic_reg_moudle.pickle")