import random
from matplotlib import pyplot as plt
import numpy as np

class Network:

  def __init__(self, layers):
    self.layers = layers
    self.bias = [np.ones((y, 1)) for y in layers[1:]]
    self.weight = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]
    dw, db =[np.zeros(w.shape) for w in self.weight], [np.zeros(b.shape) for b in self.bias]
    self.t,self.m_w,self.m_b,self.v_w,self.v_b = 0,dw,db,dw,db
    self.b, self.w, self.c, self.i =  [],[],[],[]

  def feedforward(self, a):
    for b, w in zip(self.bias, self.weight):
      a = simgmoid_func(np.dot(np.transpose(w), a) + b)
    return a

  def SGD(self, data, epochs, batch_size, alpha):
    data = data
    n = len(data)
    for j in range(epochs):
      random.shuffle(data)
      mini_batches = [data[k:k+batch_size] for k in range(0, n, batch_size)]
      for mini_batch in mini_batches:
        self.update(mini_batch,alpha)
      self.c.append(self.evaluate(data))
      self.i.append(j)

  def adam(self, mini_batch,alpha):
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    x = np.column_stack([batch[0] for batch in mini_batch])
    y = np.column_stack([batch[1] for batch in mini_batch])
    grad_b, grad_w = self.backprop(x, y)
    b1, b2, epsilon = 0.9, 0.999, 10e-8
    self.t += 1
    self.m_w = [b1 * m + (1 - b1) * g for m, g in zip(
    self.m_w,grad_w)]
    self.m_b = [b1 * m + (1 - b1) * g for m, g in zip(
    self.m_b,grad_b)]
    self.v_w = [b2 * v + (1 - b2) * g**2 for v, g in zip(
    self.v_w,grad_w)]
    self.v_b = [b2 * v + (1 - b2) * g**2 for v, g in zip(
    self.v_b,grad_b)]
    m_w_prev = [m/(1 - b1**self.t) for m in self.m_w]
    v_w_prev = [m/(1 - b2**self.t) for m in self.v_w]
    m_b_prev = [m/(1 - b1**self.t) for m in self.m_b]
    v_b_prev = [m/(1 - b2**self.t) for m in self.v_b]
    delta_w = [(alpha*m/(np.sqrt(v)+epsilon)) for m, v in zip(
      m_w_prev,v_w_prev)]
    delta_b = [(alpha*m/(np.sqrt(v)+epsilon)) for m, v in zip(
      m_b_prev,v_b_prev)]
    self.weight = [w-nw for w, nw in zip(self.weight, np.transpose(delta_w))]
    self.bias = [b-nb for b, nb in zip(self.bias, delta_b)]
    self.w.append(self.weight)
    self.b.append(self.bias)

  def update(self, mini_batch, alpha):
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    x = np.column_stack([batch[0] for batch in mini_batch])
    y = np.column_stack([batch[1] for batch in mini_batch])
    grad_b, grad_w = self.backprop(x, y)
    self.weight = [w-(alpha/len(mini_batch))*nw for w, nw in zip(self.weight, np.transpose(grad_w))]
    self.bias = [b-(alpha/len(mini_batch))*nb for b, nb in zip(self.bias, grad_b)]
    self.w.append(self.weight)
    self.b.append(self.bias)

  def backprop(self, x, y):
    num_layers = len(self.layers)-1
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.bias, self.weight):
      z = np.dot(np.transpose(w), activation)+b
      zs.append(z)
      activation = simgmoid_func(z)
      activations.append(activation)
    deltas = self.cost_function(activations[-1], y) * sigmoid_prime(zs[-1])
    grad_b[-1] = deltas.sum(axis=1).reshape((len(deltas), 1))
    grad_w[-1] = np.dot(deltas, activations[-2].transpose())
    for l in range(2, num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      deltas = np.dot(self.weight[-l+1].transpose(), deltas) * sp
      grad_b[-l] = deltas.sum(axis=1).reshape((len(deltas), 1))
      grad_w[-l] = np.dot(deltas, activations[-l-1].transpose())
    return (grad_b, grad_w)

  def evaluate(self, data):
    sum = 0
    for (x, y) in data:
      a = self.feedforward(x)
      sum += np.linalg.norm(y*np.log(a)+(1-y)*np.log(1-a))**2
    return sum

  def cost_function(self, output, y):
    return (output-y)/((1-output)*output)

def simgmoid_func(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return simgmoid_func(z)*(1-simgmoid_func(z))

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

from sklearn.datasets import make_classification

X, Y = make_classification(n_samples=100,n_features=2,class_sep=2,n_redundant=0)
X1, X2 = np.transpose(X)[0], np.transpose(X)[1]
X1, X2 = NormalizeData(X1),NormalizeData(X2)
X = [(X1, X2) for X1, X2 in zip(X1,X2)]

training_data = [[x,y] for x,y in zip(X,Y)]
net = Network([2,1])
data_size = len(training_data)
epochs = 1000
net.SGD(training_data, epochs, data_size,0.01)

y_pred = np.array([net.feedforward(i) for i in X])
w1 = net.weight[0][0][0]
w2 = net.weight[0][1][0]
b = net.bias[0][0][0]

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

plt.subplot(122)
plt.grid(True)
plt.ylabel('Y')
plt.xlabel('X')
plt.scatter(X1, X2, c=Y)
plt.plot(np.arange(0,1.1,0.1), -w1/w2*np.arange(0,1.1,0.1) - b/w2)
plt.ylim(-0.1,1.1)
plt.legend()