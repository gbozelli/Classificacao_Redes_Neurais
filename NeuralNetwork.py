import numpy as np
import random as rd
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import time
import warnings

warnings.filterwarnings('ignore')

class NeuralNetwork(object):

  def __init__(self, layers, optimizer):
    self.n_layers = len(layers)-1
    self.layers = layers
    w = [np.random.rand(layers[i],layers[i+1]) for i in range(self.n_layers)]
    b = [np.random.rand(1,layers[i+1]) for i in range(self.n_layers)]
    self.w = w
    self.b = b
    self.n_inputs = layers[0]
    if optimizer == 'adam':
      dw, db = 0*w, 0*b
      self.t,self.momentum1_w,self.momentum1_b,self.momentum2_w,self.momentum2_b = 0,dw,db,dw,db

  def product(self, x, w, b):
    z = np.array(np.matmul(x,w) + b)
    return z

  def activationFunction(self, z):
    a = np.array((1/(1+np.exp(-1*(z)))))
    return a

  def activationDerivative(self, z):
    return self.activationFunction(z)*(
    1-self.activationFunction(z))

  def costFunction(self, y_calculado, y_previsto):
    return (y_calculado - y_previsto)**2

  def costFunctionCalculated(self, y_predicted, y):
    return sum((y_predicted - y)**2)

  def multInputs(self,ref):
    Y_bin = []
    for i in range(self.layers[-1]):
      if i == ref:
        Y_bin.append(1)
      else:
        Y_bin.append(0)
    return Y_bin

  def outputs(self, x, w, b):
    s = self.activationFunction(
      self.product(x,w,b))
    return s

  def backPropagation(self, x, y):
    delta_b = self.b
    delta_w = self.w
    e, inputs,products = x, [x], []
    for i in range(self.n_layers):
      p = np.dot(e, self.w[i])+self.b[i]
      products.append(p)
      e = self.activationFunction(p)
      inputs.append(e)
    delta = self.costFunction(inputs[-1], y) * \
      self.activationFunction(products[-1])
    delta_b[-1] = delta
    delta_w[-1] = np.dot(inputs[-2].transpose(), delta)
    for l in range(2, self.n_layers):
      p = products[-l]
      e = self.activationFunction(p)
      delta = np.dot(delta, self.w[-l+1].transpose()) * e
      delta_b[-l] = delta
      delta_w[-l] = np.dot(delta, inputs[-l-1].transpose())
    return (delta_b, delta_w)

  def updateMiniBatch(self, learning_rate, b_gradients, w_gradients, delta_b, delta_w, batch_size):
    delta_b, delta_w += b_gradients*learning_rate/batch_size, w_gradients*learning_rate/batch_size
    return delta_b, delta_w

  def adamOptimizer(self, b_gradients, w_gradients):
    alpha, b1, b2, epsilon = 0.001, 0.9, 0.999, 10e-8
    t += 1
    momentum1_w = [b1 * m + (1 - b1) * g for m, g in zip(momentum1_w,w_gradients)]
    momentum1_b = [b1 * m + (1 - b1) * g for m, g in zip(momentum1_b,b_gradients)]
    momentum2_w = [b2 * v + (1 - b2) * g**2 for v, g in zip(momentum2_w,w_gradients)]
    momentum2_b = [b2 * v + (1 - b2) * g**2 for v, g in zip(momentum2_b,b_gradients)]
    momentum1_w_prev = [m/(1 - b1**t) for m in momentum1_w]
    momentum2_w_prev = [m/(1 - b2**t) for m in momentum2_w]
    momentum1_b_prev = [m/(1 - b1**t) for m in momentum1_b]
    momentum2_b_prev = [m/(1 - b2**t) for m in momentum2_b]
    delta_w += [(alpha*m/(np.sqrt(v)+epsilon)) for m, v in zip(
            momentum1_w_prev,momentum2_w_prev)]
    delta_b += [(alpha*m/(np.sqrt(v)+epsilon)) for m, v in zip(
            momentum1_b_prev,momentum2_b_prev)]
    return delta_b, delta_w,t,momentum1_w,momentum1_b,momentum2_w,momentum2_b

  def gradientDescent(self,X,Y,n_batch,epochs):
    t,momentum1_w,momentum1_b,momentum2_w,momentum2_b = 0,self.w,self.b,self.w,self.b
    mini_batch_x = [X[k:k+n_batch] for k in range(0, int(len(X)/n_batch))]
    mini_batch_y = [Y[k:k+n_batch] for k in range(0, int(len(X)/n_batch))]
    for i in range(epochs):
      for batch_x, batch_y in zip(mini_batch_x, mini_batch_y):
        delta_b, delta_w = 0*self.b, 0*self.w
        for x,y in zip(batch_x, batch_y):
          y = self.multInputs(y)
          b_gradients, w_gradients = self.backPropagation(x, y)
          delta_b, delta_w = self.adamOptimizer(b_gradients, w_gradients)
        self.w = [w-nw for w, nw in zip(self.w, delta_w)]
        self.b = [b-nb for b, nb in zip(self.b, delta_b)]
      y = self.propagation(X)
      Y = [self.multInputs(i) for i in Y]
      print('Época ',i+1, ': ',self.costFunctionCalculated(y,Y)[0])

  def normalizeData(self, data):
    '''Normaliza os data dos conjuntos entre 0 e 1'''
    self.min = np.min(data)
    self.max = np.min(data)
    data_norm = (data-np.min(data))/(
      np.max(data)-np.min(data))
    return data_norm

  def desnormalizeData(self, data_norm):
    '''Renormaliza os data dos conjuntos entre o valor
    máximo e o valor mínimo'''
    min = self.min
    max = self.max
    data = data_norm*(max-min)+min
    return data

  def propagation(self, a):
      for b, w in zip(self.b, self.w):
          a = sigmoid(np.dot(a,w)+b)
      return a

  def test(self, x_test, y_test):
    test_results = [(np.argmax(self.propagation(x)), int(y))
                        for x, y in zip(x_test, y_test)]
    print(test_results)
    return sum(int(x == y) for x, y in test_results)/len(y_test)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

X = mnist["data"]
Y = mnist["target"]
t = int(len(Y)*0.8)
f = int(len(Y))
X_train, Y_train, X_test, Y_test = X[0:t], Y[0:t], X[t:f], Y[t:f]

r = NeuralNetwork([784,10,10], 'adam')
size_batch = 16
r.gradientDescent(X_train,Y_train,size_batch,3000)
print(r.test(X_test, Y_test))
