import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neural_network


if __name__ == '__main__':
    n = neural_network.NeuralNetwork(784, 200, 10)
    nn_w_pd = pd.read_pickle(os.path.join("module", "neural_network.pickle"))
    n.wih = nn_w_pd["wih"][0]
    n.who = nn_w_pd["who"][0]

    test_pd = pd.read_csv(os.path.join("data", "mnist_test.csv"), header=None)
    test_np = np.array(test_pd)
    # test_piece = test_np[0, 1:].reshape((28, 28))
    # print(n.query(test_np[0, 1:]))
    # plt.title("测试")
    # plt.imshow(test_piece, cmap="Greys", interpolation="None")
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.show()
    scorecard = []
    for test_piece in test_np:
        result = n.query(test_piece[1:] / 255.0 * 0.99 + 0.01)
        label = np.argmax(result)
        target = test_piece[0]
        print("预测值： %d, 正确值： %d" % (label, target))
        if label == target:
            scorecard.append(1)
        else:
            scorecard.append(0)
        pass
    scorecard = np.asarray(scorecard)
    print(scorecard.sum() / scorecard.size)
