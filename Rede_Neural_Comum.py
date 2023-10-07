import numpy as np
import random as rd
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import warnings

warnings.filterwarnings('ignore')

class Rede_Neural(object):

  def __init__(self, camadas):
    '''Inicializa a rede neural através da criação das matrizes
    de pesos e da matriz de viéses, além de especificar a quantidade
    de neurônios em cada camada'''
    self.n_camadas = len(camadas)-1
    self.camadas = camadas
    w = [np.random.rand(camadas[i],camadas[i+1]) for i in range(self.n_camadas)]
    b = [np.random.rand(1,camadas[i+1]) for i in range(self.n_camadas)]
    self.w = w
    self.b = b
    self.n_entradas = camadas[0]

  def produto_matricial(self, x, w, b):
    z = np.array(np.matmul(x,w) + b)
    return z

  def funcao_de_ativacao(self, z):
    a = np.array((1/(1+np.exp(-1*(z)))))
    return a

  def funcao_derivada(self, z):
    return self.funcao_de_ativacao(z)*(
    1-self.funcao_de_ativacao(z))

  def funcao_de_custo(self, y_calculado, y_previsto):
    custo.append(sum((y_calculado - y_previsto)**2))
    return (y_calculado - y_previsto)**2

  def multi_saidas(self,ref):
    '''Retorna a saída na forma da saída da rede neural'''
    Y_bin = []
    for i in range(self.camadas[-1]):
      if i == ref:
        Y_bin.append(1)
      else:
        Y_bin.append(0)
    return Y_bin

  def retropropagacao(self,x,y):
    '''Retorna os gradientes dos coeficientes'''
    gradientes_b = self.b
    gradientes_w = self.w
    e, entradas,produtos = x, [x], []
    for i in range(self.n_camadas):
      p = np.dot(e, self.w[i])+self.b[i]
      produtos.append(p)
      e = self.funcao_de_ativacao(p)
      entradas.append(e)
    delta = self.funcao_de_custo(entradas[-1], y) * \
      self.funcao_de_ativacao(produtos[-1])
    gradientes_b[-1] = delta
    gradientes_w[-1] = np.dot(entradas[-2].transpose(), delta)
    for l in range(2, self.n_camadas):
      p = produtos[-l]
      e = self.funcao_de_ativacao(p)
      delta = np.dot(delta, self.w[-l+1].transpose()) * e
      gradientes_b[-l] = delta
      gradientes_w[-l] = np.dot(delta, entradas[-l-1].transpose())
    return (gradientes_b, gradientes_w)

  def descida_de_gradiente(self,X,Y,n,epocas):
    '''Realiza o treinamento'''
    t = 0
    m1, m2, v1, v2 = self.w, self.b, self.w, self.b
    for i in range(epocas):
      mini_batch_x = [X[k:k+n] for k in range(0, int(len(X)/n))]
      mini_batch_y = [Y[k:k+n] for k in range(0, int(len(X)/n))]
      for batch_x, batch_y in zip(mini_batch_x, mini_batch_y):
        '''Usa os batchs para calcular os gradientes'''
        delta_w, delta_b = [], []
        for x,y in zip(batch_x, batch_y):
          y = self.multi_saidas(y)
          alpha, b1, b2, epsilon = 0.001, 0.9, 0.999, 10e-8
          gradientes_b,gradientes_w = self.retropropagacao(x,y)
          t += 1
          m1 = [b1 * m + (1 - b1) * g for m, g in zip(m1,gradientes_w)]
          m2 = [b1 * m + (1 - b1) * g for m, g in zip(m2,gradientes_b)]
          v1 = [b2 * v + (1 - b2) * g**2 for v, g in zip(v1,gradientes_w)]
          v2 = [b2 * v + (1 - b2) * g**2 for v, g in zip(v2,gradientes_b)]
          m1_estimado = [m/(1 - b1**t) for m in m1]
          v1_estimado = [m/(1 - b2**t) for m in v1]
          m2_estimado = [m/(1 - b1**t) for m in m2]
          v2_estimado = [m/(1 - b2**t) for m in v2]
          delta_w += [(alpha*m/(np.sqrt(v)+epsilon)) for m, v in zip(
            m1_estimado,v1_estimado)]
          delta_b += [(alpha*m/(np.sqrt(v)+epsilon)) for m, v in zip(
            m2_estimado,v2_estimado)]
        gradientes_w, gradientes_b = delta_w, delta_b
        self.w = [w-t for w, t in zip(self.w, gradientes_w)]
        self.b = [b-t for b, t in zip(self.b, gradientes_b)]

  def normalizar(self, dados):
    '''Normaliza os dados do conjunto entre 0 e 1'''
    self.min = np.min(dados)
    self.max = np.min(dados)
    dados_norm = (dados-np.min(dados))/(
      np.max(dados)-np.min(dados))
    return dados_norm

  def desnormalizar(self, dados_norm):
    '''Desnormaliza os dados dos conjuntos entre o valor
    máximo e o valor mínimo'''
    min = self.min
    max = self.max
    dados = dados_norm*(max-min)+min
    return dados

  def propagacao(self, a):
      for b, w in zip(self.b, self.w):
          a = self.funcao_de_ativacao(np.dot(a,w)+b)
      return a

  def teste(self, x_teste, y_teste):
    test_results = [(np.argmax(self.propagacao(x)), int(y))
                        for x, y in zip(x_teste, y_teste)]
    print(test_results)
    return sum(int(x == y) for x, y in test_results)/len(y_teste)

def grafico():
  n_batch = 8
  for k in range(1,n_batch):
    r.descida_de_gradiente(X_treino,Y_treino,n,2*k)
    iteracoes = np.arange(0,len(custo))
    plt.plot(iteracoes,custo)
    custo = []
    

custo = []
iteracoes = []
r = Rede_Neural([784,10,10])
n = 4
i = 0
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist["data"]
Y = mnist["target"]
t = int(len(Y)*0.8)
f = int(len(Y))
X_treino, Y_treino, X_teste, Y_teste = X[0:t], Y[0:t], X[t:f], Y[t:f]
r.descida_de_gradiente(X_treino,Y_treino,n,300)
print(r.teste(X_teste, Y_teste))

