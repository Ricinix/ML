import os
import seaborn as sns
from util import *


def fit(data):
    output_name = data.columns[-1]
    features = data.columns[0: -1]
    counts = {}
    possible_outputs = set(data[output_name])
    for output in possible_outputs:
        counts[output] = {}
        small_data = data[data[output_name] == output]
        counts[output]["total_count"] = len(small_data)
        for f in features:
            counts[output][f] = {}
            possible_values = set(small_data[f])
            for value in possible_values:
                counts[output][f][value] = len(small_data[small_data[f] == value])
            # for x in x_set:
            #     if x not in possible_values:
            #         counts[output][f][x] = 0
    return counts


def display_counts(counts):
    for y_count in counts.keys():
        print(str(y_count) + ':')
        for f_count in counts[y_count].keys():
            if f_count == 'total_count':
                print('total_count: ' + str(counts[y_count][f_count]))
                continue
            print(f_count + ':')
            for x_count in counts[y_count][f_count].keys():
                print(str(x_count) + ':' + str(counts[y_count][f_count][x_count]))


def bayes(counts, data_verify):
    columns = data_verify.columns[0: -1]
    y_predict = []
    y_piece = {}
    possible_y = counts.keys()
    py = {}
    sum = 0
    for y in possible_y:
        sum += counts[y]['total_count']
    for y in possible_y:
        py[y] = counts[y]['total_count'] / sum

    print(py)
    for m in range(data_verify.shape[0]):

        for y in possible_y:
            px_given_y = 1
            for column in columns:
                sum = 0
                for x in counts[y][column].keys():
                    sum += counts[y][column][x]

                try:
                    px_given_y *= (counts[y][column][data_verify[column][m]] + 1) / (sum + len(counts[y][column].keys()))
                except KeyError:
                    px_given_y *= 1 / (sum + len(counts[y][column].keys()))

            y_piece[y] = px_given_y * py[y]

        y_predict.append(max(y_piece, key=y_piece.get))

    return y_predict


def verify(result, y):
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


def main():
    data_train = sns.load_dataset("iris")
    data_train[data_train.columns.values[-1]] = (data_train[data_train.columns.values[-1]] != 1) * 1
    data_train = DataPreHandle.discrete_normalization(data_train, 4, data_train.columns.values[-1])
    verify_list = FeaturePreHandle.rand_list(data_train.shape[0], int(data_train.shape[0] / 5))
    drop_list = FeaturePreHandle.pick_complementary_list(data_train.shape[0], verify_list)
    data_verify = data_train.drop(index=drop_list)
    data_train.drop(index=verify_list, inplace=True)
    # print(data_verify)
    counts = fit(data_train)
    # display_counts(counts)
    y_predict = bayes(counts, data_verify)
    verify(np.array(y_predict), np.array(data_verify[data_verify.columns.values[-1]]))


if __name__ == '__main__':
    main()
