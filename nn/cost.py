import numpy as np

def cross_entropy(output, y):
  return (output-y)/((1.0-output)*output)

def cross_calc(output, y):
  return y*np.log(output)+(1.0-y)*np.log(1.0-output)

def rmse(output, y):
  return (output-y)

def rmse_calc(output, y):
  return (output-y)**2
