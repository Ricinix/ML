import numpy as np
import pandas as pd


path = "E:\\Programming\\python_pickle\\"
data_name = 'verification_level.pickle'
module_name = 'logistic_reg_moudle.pickle'
data_train = pd.read_pickle(path + data_name)
theta = pd.read_pickle(path + module_name)


def min_max_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - np.max(X[m])) / (np.max(X[m]) - np.min(X[m]))
    return X


def zero_mean_normalization(X):
    for m in range(X.shape[0]):
        X[m] = (X[m] - X[m].mean()) / X[m].std()
    return X


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    # 加载数据
    m = data_train.shape[0]  # 样本数量
    n = data_train.shape[1] - 1  # 特征值数量
    X = np.array(data_train)
    y = X[:, n]
    X = X[:, :n]
    # 数据归一化
    X = zero_mean_normalization(X)

    z = np.dot(X, theta)
    h = sigmoid_function(z)
    result = np.round(h).flatten()

    tp = (np.logical_and(result == 1, y == 1) * 1).sum()
    print('tp:', tp)  # 真阳性
    fp = (np.logical_and(result == 1, y == 0) * 1).sum()
    print('fp:', fp)  # 假阳性
    tn = (np.logical_and(result == 0, y == 0) * 1).sum()
    print('tn:', tn)  # 真阴性
    fn = (np.logical_and(result == 0, y == 1) * 1).sum()
    print('fn:', fn)  # 假阴性

    precision = tp / (tp + fp)  # 正确率
    print('precision:', precision)

    recall = tp / (tp + fn)  # 召回率
    print('recall:', recall)

    F_measure = 2 / (1 / precision + 1 / recall)
    print('F-measure:', F_measure)  # F-Score指标(越接近1越好)
