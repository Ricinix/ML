import os
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes_num, output_nodes, learning_rate=0.3):
        self.iNodes = input_nodes
        self.hNodes_num = hidden_nodes_num
        self.oNodes = output_nodes

        self.lr = learning_rate

        self.hidden_w = []
        self.hidden_w.append(np.random.normal(0.0, pow(self.hNodes_num[0], -0.5), (self.hNodes_num[0], self.iNodes)))
        for n in range(1, len(self.hNodes_num)):
            self.hidden_w.append(np.random.normal(0.0, pow(self.hNodes_num[n], -0.5),
                                                  (self.hNodes_num[n], self.hNodes_num[n - 1])))
        self.hidden_w.append(np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes_num[-1])))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算各层的输出值
        outputs = []
        outputs.append(self.activation_function(np.dot(self.hidden_w[0], inputs)))
        for w_i in range(1, len(self.hidden_w)):
            outputs.append(self.activation_function(np.dot(self.hidden_w[w_i], outputs[w_i - 1])))

        # 梯度下降更新权值
        errors = []
        errors.append(targets - outputs[-1])
        for w_i in range(1, len(self.hidden_w)):
            errors.append(np.dot(self.hidden_w[-w_i].T, errors[w_i - 1]))

        # delta_W = α * E * sigmoid(Ok) * (1 - sigmoid(Ok)) · Oj.T

        # print("\nhidden's shape:", self.hidden_w[0].shape, "\nerrors's shape:", errors[-1].shape,
        #       "\noutputs's shape:", outputs[0].shape, "\ninputs.T's shape:", inputs.T.shape)
        self.hidden_w[0] += self.lr * np.dot(errors[-1] * outputs[0] * (1.0 - outputs[0]), inputs.T)
        for w_i in range(1, len(self.hidden_w)):
            self.hidden_w[w_i] += self.lr * np.dot(errors[-1 - w_i] * outputs[w_i] * (1.0 - outputs[w_i]),
                                                   outputs[w_i - 1].T)

    def query(self, inputs_list):
        inputs = np.array(inputs_list)
        outputs = self.activation_function(np.dot(self.hidden_w[0], inputs))
        for w in self.hidden_w[1:]:
            outputs = self.activation_function(np.dot(w, outputs))
        return outputs


def show_image(image_series):
    image_array = np.array(image_series)[1:]
    image_array = image_array.reshape((28, 28))
    print(image_array.shape)
    plt.imshow(image_array, cmap="Greys", interpolation="None")
    plt.show()


def main(save=True, hidden_nodes=None, learning_rate=None, epochs=None):
    if epochs is None:
        epochs = 6
    if hidden_nodes is None:
        hidden_nodes = [100]
    if learning_rate is None:
        learning_rate = 0.1
    input_nodes = 784
    # hidden_nodes = [100]
    output_nodes = 10

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    train_pd = pd.read_csv(os.path.join("data", "mnist_train.csv"), header=None)
    train_np = np.array(train_pd)
    for e in range(epochs):
        time = 1
        for record in train_np:
            inputs = record[1:] / 255.0 * 0.99 + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(record[0])] = 0.99

            print("第%d次，正在训练第%d个样本" % (e + 1, time))
            n.train(inputs, targets)
            time += 1

    result = pd.DataFrame([{"w": n.hidden_w, "hidden_nodes": hidden_nodes}])
    print("结果：\n", result)
    if save:
        result.to_pickle(os.path.join("module", "neural_network.pickle"))
    print("权值矩阵的个数：", len(n.hidden_w))
    return result


if __name__ == '__main__':
    main()
