from pathlib import Path
import pickle
import seaborn as sns
from util import *


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    data_train = sns.load_dataset("iris")
    with open(Path('module/logistic_reg_module.pickle'), "rb") as f:
        module = pickle.load(f)
    keep_list = module.data['drop_list']
    drop_list = FeaturePreHandle.pick_complementary_list(data_train.shape[0], keep_list)
    data_train.drop(index=drop_list, inplace=True)
    print("样本总数为: %d" % data_train.shape[0])
    theta = module.data['theta']
    data_train = DataPreHandle.str2num(data_train)
    # 加载数据
    X = np.array(data_train)
    y = (X[:, -1] != 0) * 1
    X = X[:, :-1].astype(float)
    # 数据归一化
    X = DataPreHandle.zero_mean_normalization(X)

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
