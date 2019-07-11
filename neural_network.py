import os
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.3):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        # self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算各层的输出值
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 梯度下降更新权值
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        # delta_W = α * E * sigmoid(Ok) * (1 - sigmoid(Ok)) · Oj.T
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list)
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def show_image(image_series):
    image_array = np.array(image_series)[1:]
    image_array = image_array.reshape((28, 28))
    print(image_array.shape)
    plt.imshow(image_array, cmap="Greys", interpolation="None")
    plt.show()


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.1

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # print(n.query([1.0, 0.5, -1.5]))

    train_pd = pd.read_csv(os.path.join("data", "mnist_train.csv"), header=None)
    # show_image(train_pd.iloc[0])
    train_np = np.array(train_pd)
    # scaled_input = train_np[:, 1:] / 255.0 * 0.99 + 0.01
    # print(scaled_input.shape)
    # targets = np.zeros(output_nodes) + 0.01
    # targets[int(train_pd[0][0])] = 0.99
    # print(targets)
    epochs = 6
    for e in range(epochs):
        for record in train_np:
            inputs = record[1:] / 255.0 * 0.99 + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(record[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass
    result = pd.DataFrame([{"wih": n.wih, "who": n.who}])
    print(result)
    result.to_pickle(os.path.join("module", "neural_network.pickle"))
