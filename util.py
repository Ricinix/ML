import numpy as np
import pandas as pd
import random


class DataPreHandle:

    # min-max标准化(线性标准化)
    @staticmethod
    def min_max_normalization(X: pd.DataFrame):
        for n in range(X.shape[1]):
            X[:, n] = (X[:, n] - np.min(X[:, n])) / (np.max(X[:, n]) - np.min(X[:, n]))
        return X

    # z-score 标准化(正态分布标准化)
    @staticmethod
    def zero_mean_normalization(X: pd.DataFrame):
        for n in range(X.shape[1]):
            X[:, n] = (X[:, n] - X[:, n].mean()) / X[:, n].std()
        return X

    # 把连续值分成10段的离散值
    @staticmethod
    def discrete_normalization(data: pd.DataFrame, k: int, *args):
        columns = data.columns[0: -1]
        for column in columns:
            if column in args:
                continue
            temp = (data[column].max() - data[column].min()) / k
            min = data[column].min()
            t = 0
            for i in range(1, k):
                t += (data[column] >= i * temp + min) * 1
            data[column] = t
            # data[column] = (data[column] >= 9 * temp + min) * 1 +\
            #     (data[column] >= 8 * temp + min) * 1 +\
            #     (data[column] >= 7 * temp + min) * 1 +\
            #     (data[column] >= 6 * temp + min) * 1 +\
            #     (data[column] >= 5 * temp + min) * 1 +\
            #     (data[column] >= 4 * temp + min) * 1 +\
            #     (data[column] >= 3 * temp + min) * 1 +\
            #     (data[column] >= 2 * temp + min) * 1 +\
            #     (data[column] >= 1 * temp + min) * 1
        return data

    # 将df中的字符串更改为离散的数字
    @staticmethod
    def str2num(data: pd.DataFrame, cols=None):
        if cols is None:
            cols = [data.columns.values[-1]]
        for col in cols:
            values = list(set(data[col]))
            values.sort()
            for value, i in zip(values, range(len(values))):
                data[col].replace(value, i, inplace=True)
        return data


class FeaturePreHandle:

    # 随机生成一个序列
    @staticmethod
    def rand_list(ran: int, size: int):
        r_list = [x for x in range(ran)]
        for i in range(len(r_list)-1, 0, -1):
            j = random.randint(0, i - 1)
            temp = r_list[i]
            r_list[i] = r_list[j]
            r_list[j] = temp
        return r_list[:size]

    # 获取一个序列的补集
    @staticmethod
    def pick_complementary_list(total: int, sub: list):
        total_list = [x for x in range(total)]
        for num in sub:
            total_list.remove(num)
        return total_list


class ModuleData:

    def __init__(self, **data):
        self.data = data
