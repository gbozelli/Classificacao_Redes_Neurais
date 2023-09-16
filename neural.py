import numpy as np
import random as rd
from std import rand_cluster

class Rede_Neural(object):

  def __init__(self, camadas):
    '''Inicializa a rede neural através da criação das matrizes
    de pesos e da matriz de viéses'''
    self.n_camadas = len(camadas)-1
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
    return (y_calculado - y_previsto)

  def saida(self, x, w, b):
    s = self.funcao_de_ativacao(
      self.produto_matricial(x,w,b))
    return s
  
  def descida_de_gradiente(self, x, y, iteracoes, taxa_de_aprendizado):
    soma_b = [np.zeros(b.shape) for b in self.b]
    soma_w = [np.zeros(w.shape) for w in self.w]
    for i in range(iteracoes):
      rd.shuffle(x)
      delta_b, delta_w = self.retropropagacao(x,y)
      soma_b = [nb+dnb for nb, dnb in zip(soma_b, delta_b)]
      soma_w = [nw+dnw for nw, dnw in zip(soma_w, delta_w)]
    self.w = [w-(taxa_de_aprendizado/self.n_entradas)*nw
                      for w, nw in zip(self.w, soma_w)]
    self.b = [b-(taxa_de_aprendizado/self.n_entradas)*nb
                      for b, nb in zip(self.b, soma_b)]

  def retropropagacao(self,x,y):
    delta_b = [np.zeros(b.shape) for b in self.b]
    delta_w = [np.zeros(w.shape) for w in self.w]
    e, entradas,produtos = x, [x], []
    for i in range(self.n_camadas):
      p = np.dot(e, self.w[i])+self.b[i]
      produtos.append(p)
      e = self.funcao_de_ativacao(p)
      entradas.append(e)
    delta = self.funcao_de_custo(entradas[-1], y) * \
            self.funcao_de_ativacao(produtos[-1])
    delta_b[-1] = delta
    delta_w[-1] = np.dot(entradas[-2].transpose(), delta)
    for l in range(2, self.n_camadas):
      p = produtos[-l]
      e = self.funcao_de_ativacao(p)
      delta = np.dot(delta, self.w[-l+1].transpose()) * e
      delta_b[-l] = delta
      delta_w[-l] = np.dot(delta, entradas[-l-1].transpose())
    return (delta_b, delta_w)

  def treinamento(self, x,y,taxa_de_aprendizado,iteracoes):
    self.descida_de_gradiente(x,y,iteracoes,taxa_de_aprendizado)
  
  def normalizar(self, dados):
    '''Normaliza os dados dos conjuntos entre 0 e 1'''
    self.min = np.min(dados)
    self.max = np.min(dados)
    dados_norm = (dados-np.min(dados))/(
      np.max(dados)-np.min(dados))
    return dados_norm
  
  def desnormalizar(self, dados_norm):
    '''Renormaliza os dados dos conjuntos entre o valor
    máximo e o valor mínimo'''
    min = self.min
    max = self.max
    dados = dados_norm*(max-min)+min
    return dados

  def teste(self, x):
    saida = self.saida(x,self.w[0],self.b[0])
    for i in range(1,self.n_camadas):
      saida = self.saida(saida,self.w[i],self.b[i])
    return saida

r = Rede_Neural([764,20,10,10])

