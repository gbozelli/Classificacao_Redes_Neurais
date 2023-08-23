import pandas as pd
import numpy as np
from numpy import transpose as t
from std import rand_cluster
from matplotlib import pyplot as plt

def dotProduct(x,w,b):
  z = np.matmul(x,w) + b
  return np.array(z)

def activationFunction(x):
  a = 1/(1+np.exp(-1*(x)))
  return np.array(a)

def derivativeFunction(x):
  return activationFunction(x)*(
    1-activationFunction(x))

def neuronOutput(x,w,b):
  z = dotProduct(x,w,b)
  a = activationFunction(z)
  return a

def costFunction(a,y):
  c = (sum((a-y)**2))
  return c

def gradientVector(activation,activation2,z,y):
  a,a2,b = [], [], []
  a.append(activation)
  a2.append(activation2)
  b.append(z)
  a = np.array(t(a))
  b = np.array(b)
  a2 = np.array(a2)
  gc = np.matmul(a,(derivativeFunction(b)*(a2-y)))
  return gc

def neuralTraining(x,y):
  w1 = np.array(np.random.random(
    (n_inputs,n_neurons)))
  w2 = np.array(np.random.random(
    (n_neurons,n_outputs)))
  bias1 = np.array(n_neurons)
  bias2 = np.array(n_outputs)
  steps = 100
  for i in range(steps):
    a = neuronOutput(x,w1,bias1)
    z = dotProduct(a,w2,bias2)
    activation = neuronOutput(x,w1,bias1)
    activation2 = neuronOutput(activation,w2,bias2)
    gc = gradientVector(activation,activation2,z,y)
    w2 += gc
    ym = activation
    z = dotProduct(x,w1,bias1)
    activation = x
    activation2 = neuronOutput(activation,w1,bias1)
    gc = gradientVector(activation,activation2,z,ym)
    w1 += gc
  return w1,w2,bias1,bias2

def test(x,w1,w2,y):
  neuronOutput

n_inputs = 2
n_outputs = 2
n_neurons = 4

x1 = np.array(np.transpose(rand_cluster(2,13,0.75,0.75)))
w1, w2,b1,b2 = neuralTraining(x1[0],x1[1])
nx = neuronOutput(neuronOutput(x1,w1,b1),w2,b2)
print(nx)