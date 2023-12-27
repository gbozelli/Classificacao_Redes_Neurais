import numpy as np
import nn as nn
from matplotlib import pyplot as plt

N = 100
center = 5
sigma = 20
sigma_n = 0.1
A = 5
B = 12
X = center+sigma*np.random.rand(N,1)
n = sigma_n*np.random.randn(N,1)
Y = A*X+B+n

training_data = [[x,y] for x,y in zip(X,Y)]
net = nn.Network([1,1], 'adam', ['linear'])
data_size = len(training_data)
epochs = 100
net.train(training_data, epochs, data_size, 10)

y_pred = np.array([net.feedforward(i) for i in X])

plt.figure(figsize=(13,5))
plt.subplot(121)
plt.plot(net.iterations,net.train_cost, color='black',  label='Custo de treino')
plt.plot(net.iterations,net.test_cost, color='black',  label='Custo de teste')
plt.grid(True)
plt.ylabel('Métricas de treino')
plt.xlabel('Época')
plt.xlim(0,100)
plt.title('Mínimo erro quadrático')
plt.legend()
plt.subplot(122)
plt.grid(True)
plt.ylabel('Y')
plt.xlabel('X')
plt.scatter(np.transpose(X)[0],Y, color='red',label='Dados')
plt.plot(np.sort(np.transpose(X)[0]),np.sort(np.transpose(y_pred)[0][0]), color='black',label='Função')
plt.legend()
plt.show()