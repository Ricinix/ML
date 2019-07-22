from pathlib import Path
import pickle
import seaborn as sns
from util import *


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
    data_train = sns.load_dataset('iris')
    # 随机选组 m / k 个最为测试集
    dr_l = FeaturePreHandle.rand_list(data_train.shape[0], int(data_train.shape[0] / 5))
    data_train.drop(index=dr_l)
    # 将字符串转换为数字
    data_train = DataPreHandle.str2num(data_train)
    np_data = np.array(data_train)
    X = np_data[:, :-1].astype(float)
    y = (np_data[:, -1] != 0) * 1
    # 数据归一化
    X = DataPreHandle.zero_mean_normalization(X)
    alpha = 0.01
    theta = logistic_reg(alpha, X, y, max_iterations=100000)
    module = ModuleData(theta=theta, drop_list=dr_l)
    module = pickle.dumps(module)
    with open(Path('module/logistic_reg_module.pickle'), "wb") as f:
        f.write(module)
