import numpy as np

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))

def linear(z):
  return z

def linear_prime(z):
  return np.ones(z.shape)