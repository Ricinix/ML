import numpy as np
import pandas as pd


path = "E:\\Programming\\python_pickle\\"
name = 'train_level.pickle'


def min_max_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - np.max(X[m])) / (np.max(X[m]) - np.min(X[m]))
    # sum = X.sum(axis = 1)
    # sum = sum.repeat(X.shape[1]).reshape(X.shape)
    # print(sum)
    # return X / sum
    return X


def zero_mean_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - X[m].mean()) / X[m].std()
    return X


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


def cost_function(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


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
# def h(X):
#     gx = -X[0:n].dot(theta.T)
#     ans =0.0
#     try:
#         ans = 1/(1+np.exp(gx))
#     except OverflowError:
#         ans = 0.00000000000001
#     if ans == 1.0:
#         ans = 0.99999999999999
#     return ans
#
#
# def l():
#     fun = 0.0
#     for j in range(1,m):
#             fun += (np_data[j][y] * np.log(h(np_data[j])) + (1-np_data[j][y]) * np.log(1-h(np_data[j])))
#
#     return fun
#
#
# def getTr(i):
#     tr=0.0
#     for j in range(m):
#         tr+=(np_data[j][y] - h(np_data[j])) * np_data[j][i]
#     return tr


if __name__ == "__main__":
    # while(True):
    #     fun = 0
    #     fun_new = 0
    #     for i in range(n):
    #         theta_tr[i] = getTr(i)
    #     theta = theta + theta_tr
    #     fun_new=l()
    #     print("l(theta) : ",fun_new)
    #     if fun - fun_new < eplise:
    #         break
    data_train = pd.read_pickle(path + name)
    np_data = np.array(data_train)
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np_data[:, 0:n]
    y = np_data[:, n]
    X = zero_mean_normalization(X)

    alpha = 0.1
    theta = logistic_reg(alpha, X, y, max_iterations=100000)
    print(theta)
    df = pd.DataFrame(theta)
    df.to_pickle(path + "logistic_reg_moudle.pickle")