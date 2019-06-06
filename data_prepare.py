import numpy as np
import pandas as pd

path = ".\\data\\"
name = 'verification_data.csv'
data_train = pd.read_csv(path + name)
print(data_train.columns.values)


def pre_data_channel():
    is_discrete = True
    data_train['data_channel'] = 0
    data_train['data_channel'] += (data_train['data_channel_is_lifestyle'] == 1) * 1
    data_train.drop(columns='data_channel_is_lifestyle', inplace=True)

    if (data_train['data_channel'] * data_train['data_channel_is_entertainment']).sum() != 0:
        is_discrete = False
    data_train['data_channel'] += (data_train['data_channel_is_entertainment'] == 1) * 2
    data_train.drop(columns='data_channel_is_entertainment', inplace=True)

    if (data_train['data_channel'] * data_train['data_channel_is_bus']).sum() != 0:
        is_discrete = False
    data_train['data_channel'] += (data_train['data_channel_is_bus'] == 1) * 3
    data_train.drop(columns='data_channel_is_bus', inplace=True)

    if (data_train['data_channel'] * data_train['data_channel_is_socmed']).sum() != 0:
        is_discrete = False
    data_train['data_channel'] += (data_train['data_channel_is_socmed'] == 1) * 4
    data_train.drop(columns='data_channel_is_socmed', inplace=True)

    if (data_train['data_channel'] * data_train['data_channel_is_tech']).sum() != 0:
        is_discrete = False
    data_train['data_channel'] += (data_train['data_channel_is_tech'] == 1) * 5
    data_train.drop(columns='data_channel_is_tech', inplace=True)

    if (data_train['data_channel'] * data_train['data_channel_is_world']).sum() != 0:
        is_discrete = False
    data_train['data_channel'] += (data_train['data_channel_is_world'] == 1) * 6
    data_train.drop(columns='data_channel_is_world', inplace=True)

    return is_discrete


def pre_weekday():
    is_discrete = True
    data_train['weekday'] = 0
    data_train['weekday'] += (data_train['weekday_is_monday'] == 1) * 1
    data_train.drop(columns='weekday_is_monday', inplace=True)

    if (data_train['weekday'] * data_train['weekday_is_tuesday']).sum() != 0:
        is_discrete = False
    data_train['weekday'] += (data_train['weekday_is_tuesday'] == 1) * 2
    data_train.drop(columns='weekday_is_tuesday', inplace=True)

    if (data_train['weekday'] * data_train['weekday_is_wednesday']).sum() != 0:
        is_discrete = False
    data_train['weekday'] += (data_train['weekday_is_wednesday'] == 1) * 3
    data_train.drop(columns='weekday_is_wednesday', inplace=True)

    if (data_train['weekday'] * data_train['weekday_is_thursday']).sum() != 0:
        is_discrete = False
    data_train['weekday'] += (data_train['weekday_is_thursday'] == 1) * 4
    data_train.drop(columns='weekday_is_thursday', inplace=True)

    if (data_train['weekday'] * data_train['weekday_is_friday']).sum() != 0:
        is_discrete = False
    data_train['weekday'] += (data_train['weekday_is_friday'] == 1) * 5
    data_train.drop(columns='weekday_is_friday', inplace=True)

    if (data_train['weekday'] * data_train['weekday_is_saturday']).sum() != 0:
        is_discrete = False
    data_train['weekday'] += (data_train['weekday_is_saturday'] == 1) * 6
    data_train.drop(columns='weekday_is_saturday', inplace=True)

    if (data_train['weekday'] * data_train['weekday_is_sunday']).sum() != 0:
        is_discrete = False
    data_train['weekday'] += (data_train['weekday_is_sunday'] == 1) * 7
    data_train.drop(columns='weekday_is_sunday', inplace=True)

    data_train.drop(columns='is_weekend', inplace=True)
    return is_discrete


def pre_for_shares():
    data_train['target_shares'] = data_train['shares']
    data_train.drop(columns=['shares', 'level', 'url'], inplace=True)


def pre_for_level():
    data_train['target_level'] = data_train['level']
    data_train.drop(columns=['shares', 'level', 'url'], inplace=True)


if __name__ == '__main__':

    if pre_data_channel() and pre_weekday():
        print("部分离散特征值已线性连续化")
    # pre_for_level()
    pre_for_shares()
    data_train.to_pickle(path + 'verification_shares.pickle')

