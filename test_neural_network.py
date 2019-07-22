import neural_network
import verify_neural_network
import seaborn as sns
import matplotlib.pyplot as plt


def epochs_test():
    scores = []
    test_e = [x for x in range(3, 21)]
    for e in test_e:
        result = neural_network.main(False, [100], 0.3, e)
        scores.append(verify_neural_network.main(result))
    # plt.plot(test_e, scores)
    # plt.ylabel("score")
    # plt.xlabel("epochs")
    # plt.title("epochs test")
    sns.relplot(x=test_e, y=scores, kind='line')
    plt.show()


def learning_rate_test():
    pass


if __name__ == '__main__':
    epochs_test()
