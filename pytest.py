import numpy as np
import pandas as pd

path = "E:\\Programming\\python_pickle\\"
name = 'train_data.pickle'
data_train = pd.read_pickle(path + name)
print(data_train.head())
