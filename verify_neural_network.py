import os
import pandas as pd
import numpy as np
import neural_network


def main(nn_w_pd=None):
    if nn_w_pd is None:
        nn_w_pd = pd.read_pickle(os.path.join("module", "neural_network.pickle"))
    n = neural_network.NeuralNetwork(784, nn_w_pd["hidden_nodes"][0], 10)
    n.hidden_w = nn_w_pd["w"][0]

    test_pd = pd.read_csv(os.path.join("data", "mnist_test.csv"), header=None)
    test_np = np.array(test_pd)
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
    score = scorecard.sum() / scorecard.size
    print(score)
    return score


if __name__ == '__main__':
    main()
