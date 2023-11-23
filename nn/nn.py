import numpy as np

import activation as act
import cost as c
import optmizer as opt

class Network:

  def __init__(self, layers, optmizer, activations):

    self.train_cost = []
    self.test_cost = []
    self.iterations = []
    self.activactions = []
    self.activactions_prime = []
    self.layers = layers
    self.num_layers = len(self.layers)-1
    self.bias = [np.ones((y, 1)) for y in layers[1:]]
    self.weight = [np.random.rand(x, y) for x, y in zip(layers[:-1], layers[1:])]

    if optmizer == 'adam':
      dw, db =[np.zeros(w.shape) for w in self.weight], [np.zeros(b.shape) for b in self.bias]
      self.t,self.m_w,self.m_b,self.v_w,self.v_b = 0,dw,db,dw,db
      self.optmizer = opt.optmizer.adam
    elif optmizer == 'sgd':
      self.optmizer = opt.optmizer.sgd
    
    for a in activations:
      if a == 'linear':
        self.activactions.append(act.linear)
        self.activactions_prime.append(act.linear_prime)
      elif a == 'sigmoid':
        self.activactions.append(act.sigmoid)
        self.activactions_prime.append(act.sigmoid_prime)
    
    if activations[-1] == 'linear':
      self.cost_function = c.rmse
      self.cost_calc = c.rmse_calc
    elif activations[-1] == 'sigmoid':
      self.cost_function = c.cross_entropy
      self.cost_calc = c.cross_calc

  def feedforward(self, a):

    i = 0
    for b, w in zip(self.bias, self.weight):
      a = self.activactions[i](np.dot(np.transpose(w), a) + b)
      i += 1
    return a

  def train(self, train_data, epochs, batch_size, alpha, test_data=None):

    if test_data == None:
      test_data = train_data
    n = len(train_data)
    for i in range(epochs):
      np.random.shuffle(train_data)
      mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
      for mini_batch in mini_batches:
        self.optmizer(self, mini_batch, alpha)
      self.test_cost.append(self.evaluate(test_data))
      self.train_cost.append(self.evaluate(train_data))
      self.iterations.append(i)
      print(self.weight)

  def backprop(self, x, y):
    
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    activation = x
    activations = [x]
    zs = []
    i = 0
    for b, w in zip(self.bias, self.weight):
      z = np.dot(np.transpose(w), activation)+b
      zs.append(z)
      activation = self.activactions[i](z)
      activations.append(activation)
      i += 1
    deltas = self.cost_function(activations[-1], y) * self.activactions_prime[-1](zs[-1])
    grad_b[-1] = deltas.sum(axis=1).reshape((len(deltas), 1))
    grad_w[-1] = np.dot(deltas, activations[-2].transpose())
    i = 2
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = self.activactions_prime[-i](z)
      deltas = np.dot(self.weight[-l+1].transpose(), deltas) * sp
      grad_b[-l] = deltas.sum(axis=1).reshape((len(deltas), 1))
      grad_w[-l] = np.dot(deltas, activations[-l-1].transpose())
      i += 1
    return (grad_b, grad_w)

  def evaluate(self, data):
    
    sum = 0
    for (x, y) in data:
      a = self.feedforward(x)
      sum += np.linalg.norm(self.cost_calc(a, y))
    return sum
  
  def cost_parameters(self):
    return self.test_cost, self.train_cost

def normalize(data):
  norm_data = [(x - np.min(x)) / (np.max(x) - np.min(x)) for x in data]
  return norm_data
