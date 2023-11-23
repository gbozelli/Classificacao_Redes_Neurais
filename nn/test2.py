import numpy as np
import nn as net
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

X, Y = make_classification(n_samples=100,n_features=2,class_sep=2,n_redundant=0,n_informative=2,n_clusters_per_class=1)
X1, X2 = np.transpose(X)[0], np.transpose(X)[1]
X1, X2 = NormalizeData(X1),NormalizeData(X2)
X = [(X1, X2) for X1, X2 in zip(X1,X2)]

training_data = [[x,y] for x,y in zip(X,Y)]

Net = net.Network([2,1], 'sgd', ['sigmoid'])
data_size = len(training_data)
epochs = 100
Net.train(training_data, epochs, data_size,10)

plt.subplot(121)
plt.plot(Net.iterations,Net.train_cost, color='black',  label='Custo de treino')
plt.plot(Net.iterations,Net.test_cost, color='black',  label='Custo de teste')
plt.grid(True)
plt.ylabel('Métricas de treino')
plt.xlabel('Época')
plt.title('Mínimo erro quadrático')
plt.legend()

w1 = Net.weight[0][0][0]
w2 = Net.weight[0][1][0]
b = Net.bias[0][0][0]

plt.subplot(122)
plt.grid(True)
plt.ylabel('Y')
plt.xlabel('X')
plt.scatter(X1, X2, c=Y)
plt.plot(np.arange(0,1.1,0.1), -w1/w2*np.arange(0,1.1,0.1) - b/w2)
plt.ylim(-0.1,1.1)
plt.legend()

plt.show()