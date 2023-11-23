import numpy as np

class optmizer:

  def adam(self, mini_batch,  alpha):
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

  def sgd(self, mini_batch, alpha):
    grad_b = [np.zeros(b.shape) for b in self.bias]
    grad_w = [np.zeros(w.shape) for w in self.weight]
    x = np.column_stack([batch[0] for batch in mini_batch])
    y = np.column_stack([batch[1] for batch in mini_batch])
    grad_b, grad_w = self.backprop(x, y)
    self.weight = [w-(alpha/len(mini_batch))*nw for w, nw in zip(
      self.weight, np.transpose(grad_w))]
    self.bias = [b-(alpha/len(mini_batch))*nb for b, nb in zip(
      self.bias, grad_b)]
