import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import baseline
import allZero
import mse
import learning_rate
import regularization


def main():
    # 读取数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 各种方法
    epoch_list, baseline_accuracy_list = baseline.baseLine(mnist)
    epoch_list, allZero_accuracy_list = allZero.allZero(mnist)
    epoch_list, mse_accuracy_list = mse.mse(mnist)
    epoch_list, learning_rate_accuracy_list = learning_rate.learning_rate(mnist)
    epoch_list, regularization_accuracy_list = regularization.regularization(mnist)

    # 性能曲线
    plt.plot(epoch_list, baseline_accuracy_list, label="baseline")
    plt.plot(epoch_list, allZero_accuracy_list, label="allZero")
    plt.plot(epoch_list, mse_accuracy_list, label="mse")
    plt.plot(epoch_list, learning_rate_accuracy_list, label="learning_rate")
    plt.plot(epoch_list, regularization_accuracy_list, label="regularization")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
