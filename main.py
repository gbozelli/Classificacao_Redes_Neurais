import pandas as pd
import numpy as np

def dotProduct(x,w,b):
  z = np.matmul(x,w) + b
  return z

def activationFunction(x):
  a = 1/(1+np.exp(-(x)))
  return a

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

def gradientVector(a1,a2,z,y):
  gc = a1*derivativeFunction(z)*2*(a2-y)
  return gc

def neuralTraining(x,y):
  w = np.array()
  bias = np.array(numberLayers)
  w.append(np.zeros(numberInputs,numberNeurons))
  w.append(np.zeros(numberNeurons,numberOutputs))
  steps = 10
  for i in range(steps):
    z = dotProduct(x,w,bias)
    a1 = neuronOutput(x,w[i],bias)
    a2 = neuronOutput(a1,w[i+1],bias)
    gc = gradientVector(a1,a2,z,y)
    w += gc
  return w


numberInputs = 0
numberOutputs = 0
numberLayers = 0
numberNeurons = 0

