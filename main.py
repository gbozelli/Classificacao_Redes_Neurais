import pandas as pd
import numpy as np

def dotProduct(x,w,b):
  s = np.matmul(x,w) + b
  return s

def activationFunction(x):
  y = 1/(1+np.exp(-(x)))
  return y

def costFunction(X,YM):
  J = 1/numberInputs*(
    sum(sum((X-YM)**2)
  ))

numberInputs = 0
numberOutputs = 0
numberNeurons = 0

