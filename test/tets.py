import random
from matplotlib import pyplot as plt
import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

from sklearn.datasets import make_classification

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

X, Y = make_classification(n_samples=10,n_features=2,class_sep=10,n_redundant=0,n_informative=2,n_clusters_per_class=1)
X1, X2 = np.transpose(X)[0], np.transpose(X)[1]
X1, X2 = NormalizeData(X1),NormalizeData(X2)
X = [(X1, X2) for X1, X2 in zip(X1,X2)]

training_data = [[x,y] for x,y in zip(X,Y)]

net = Network1([2,1])
data_size = len(training_data)
epochs = 100
net.SGD(training_data, epochs, data_size,0.1)

plt.figure(figsize=(13,5))
plt.subplot(121)
plt.plot(net.i,net.c, color='black',  label='Função de custo')
plt.plot(net.i,np.array(np.transpose(net.w)[0][0]).reshape(epochs), color='blue',  label='Coeficiente W1')
plt.plot(net.i,np.array(np.transpose(net.w)[0][1]).reshape(epochs), color='red',  label='Coeficiente W2',alpha=0.6)
plt.plot(net.i,np.array(net.b).reshape(epochs), color='green',  label='Coeficiente B',alpha=0.6)
plt.grid(True)
plt.ylabel('Métricas de treino')
plt.xlabel('Época')
plt.title('Mínimo erro quadrático')
plt.legend()

w1 = net.weight[0][0][0]
w2 = net.weight[0][1][0]
b = net.bias[0][0][0]

plt.subplot(122)
plt.grid(True)
plt.ylabel('Y')
plt.xlabel('X')
plt.scatter(X1, X2, c=Y)
plt.plot(np.arange(0,1.1,0.1), -w1/w2*np.arange(0,1.1,0.1) - b/w2)
plt.ylim(-0.1,1.1)
plt.show()