from Datasets.Mnist_dataset import mnist_dataset
from Neural_Networks.IBM_neural_net_implementation import DeepNeuralNetwork


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x_train, x_val, y_train, y_val = mnist_dataset()
    dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
    dnn.train(x_train, y_train, x_val, y_val)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
