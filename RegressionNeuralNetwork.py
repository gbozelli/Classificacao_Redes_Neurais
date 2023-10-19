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

  def feedforward(self, a):
    for b, w in zip(self.bias, self.weight):
      a = identity_func(np.dot(np.transpose(w), a) + b)
    return a

  def SGD(self, data, epochs, batch_size, alpha):
    data = data
    n = len(data)
    for j in range(epochs):
      random.shuffle(data)
      mini_batches = [data[k:k+batch_size] for k in range(0, n, batch_size)]
      for mini_batch in mini_batches:
        self.adam(mini_batch)
      c.append(self.evaluate(data))
      i.append(j)

  def adam(self, mini_batch):
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    x = np.column_stack([batch[0] for batch in mini_batch])
    y = np.column_stack([batch[1] for batch in mini_batch])
    grad_b, grad_w = self.backprop(x, y)
    alpha, b1, b2, epsilon = 10, 0.9, 0.999, 10e-8
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
    self.weight = [w-nw for w, nw in zip(
      self.weight, np.transpose(delta_w))]
    self.bias = [b-nb for b, nb in zip(self.bias, delta_b)]
    w.append(self.weight)
    b.append(self.bias)

  def update(self, mini_batch, alpha):
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    x = np.column_stack([batch[0] for batch in mini_batch])
    y = np.column_stack([batch[1] for batch in mini_batch])
    grad_b, grad_w = self.backprop(x, y)
    self.weight = [w-(alpha/len(mini_batch))*nw for w, nw in zip(
      self.weight, np.transpose(grad_w))]
    self.bias = [b-(alpha/len(mini_batch))*nb for b, nb in zip(
      self.bias, grad_b)]
    w.append(self.weight)
    b.append(self.bias)

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
      activation = identity_func(z)
      activations.append(activation)
    deltas = self.cost_function(activations[-1], y) * identity_prime(zs[-1])
    grad_b[-1] = deltas.sum(axis=1).reshape((len(deltas), 1))
    grad_w[-1] = np.dot(deltas, activations[-2].transpose())
    for l in range(2, num_layers):
      z = zs[-l]
      sp = identity_prime(z)
      deltas = np.dot(self.weight[-l+1].transpose(), deltas) * sp
      grad_b[-l] = deltas.sum(axis=1).reshape((len(deltas), 1))
      grad_w[-l] = np.dot(deltas, activations[-l-1].transpose())
    return (grad_b, grad_w)

  def evaluate(self, data):
    sum = 0
    for (x, y) in data:
      sum += np.linalg.norm(self.feedforward(x)-y)**2
    return sum

  def cost_function(self, output, y):
    return (output-y)

def identity_func(z):
  return z

def identity_prime(z):
  return np.ones(z.shape)

c = []
w = []
b = []
i = []