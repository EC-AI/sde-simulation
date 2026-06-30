import numpy as np

def euler_maruyama(x0, z, T = 1,  a = lambda x, t: 0, b = lambda x, t: 1):
  """
  Numerical solution of a Stochastic Differential Equation using Euler-Maruyama method.

  Args:
    x0 (float): The initial value of the process.
    z (np.ndarray): Array of simulated standard normal random variates to generate the increments.
    T (float): End time of the simulation (default 1).
    a (function): The drift function a(x, t) (default is lambda x, t: 0).
    b (function): The diffusion function b(x, t) (default is lambda x, t: 1).

  Returns:
    np.ndarray: Simulated trajectory of the process.
  """
  n = z.size
  h = T/n
  sqrth = np.sqrt(h)
  t = np.linspace(0, T, n+1)

  X = np.zeros(n+1)
  X[0] = x0

  for i in range(1, n+1):
    X[i] = X[i-1] + a(X[i-1], t[i-1]) * h + z[i-1] * b(X[i-1], t[i-1]) * sqrth

  return X


def milstein(x0, z, T=1, a=lambda x, t: 0, b=lambda x, t: 1, dbdx=None):
  """
  Numerical solution of an SDE using the Milstein method.
  Args:
      x0 (float): The initial value of the process.
      z (np.ndarray): Array of simulated standard normal random variates to generate the increments.
      T (float): End time of the simulation.
      a (function): The drift function a(x, t).
      b (function): The diffusion function b(x, t).
      dbdx (function): The derivative of the diffusion function with respect to x, db/dx(x, t).
  Returns:
      np.ndarray: Simulated trajectory of the process.
  """
  n = z.size
  h = T / n
  sqrth = np.sqrt(h)
  t = np.linspace(0, T, n + 1)

  X = np.zeros(n + 1)
  X[0] = x0

  for i in range(1, n + 1):
      dW = z[i - 1] * sqrth
      X[i] = (
        X[i - 1]
        + a(X[i - 1], t[i - 1]) * h
        + b(X[i - 1], t[i - 1]) * dW
        + 0.5 * b(X[i - 1], t[i - 1]) * dbdx(X[i - 1], t[i - 1]) * (dW ** 2 - h)
      )
  return X