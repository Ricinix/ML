import numpy as np
import pandas as pd
import math

path = "E:\\Programming\\python_pickle\\"
name = 'train_level.pickle'
data_train = pd.read_pickle(path + name)

np_data = np.array(data_train)

eplise = 0.000001
m = data_train.shape[0]  # 样本数量
n = data_train.shape[1]-1     # 特征值数量
y = n
theta = np.random.rand(n)
theta_tr = np.random.rand(n)


def h(X):
    gx = -X[0:n].dot(theta.T)
    ans =0.0
    try:
        ans = 1/(1+math.exp(gx))
    except OverflowError:
        ans = 0.0000001
    if ans == 1.0:
        ans = 0.9999999
    return ans


def l():
    fun = 0.0
    for j in range(1,m):
            fun += np_data[j][y] * math.log(h(np_data[j])) + (1-np_data[j][y]) * math.log(1-h(np_data[j]))
        
    return fun


def getTr(i):
    tr=0.0
    for j in range(m):
        tr+=(np_data[j][y] - h(np_data[j])) * np_data[j][i]
    return tr

if __name__ == "__main__":
    while(True):
        fun = 0
        fun_new = 0
        for i in range(n):
            theta_tr[i] = getTr(i)
        theta = theta + theta_tr
        fun_new=l()
        print("l(theta) : ",fun_new)
        if fun - fun_new < eplise:
            break
