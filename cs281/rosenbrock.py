import matplotlib.pyplot as plt
from scipy.optimize import check_grad
import numpy as np
from numpy import dot, log, outer, exp

np.set_printoptions(suppress=True)
_lambda = 1

def shuffle(X, y):
    idx = np.arange(y.size)
    return X[idx,:], y[idx]

def split(X, y, r=.8):
    idx = r*y.size
    return X[:idx,:], y[:idx], X[idx:,:], y[idx:]

def log_likelihood(theta, s_k, y_k, a_k):
    theta = np.exp(theta)
    #Unroll
    W = np.reshape(theta[:], (s_k.size,s_k.size))
    #Predicted mean
    mu_k = dot(s_k, dot(W, y_k))/dot(y_k, y_k)
    #Likelihood
    x = .5*(a_k - mu_k)**2/(mu_k**2*a_k)
    f = .5*log(_lambda) - _lambda*x
    #Gradient
    dtheta =  _lambda*(a_k - mu_k)/(a_k*dot(y_k,y_k)) * (1/mu_k**2 + (a_k - mu_k)/(mu_k**3)) * np.ravel(outer(s_k, y_k))
    return f, theta*dtheta

def check_likelihood():
    sk = np.random.rand(10)
    yk = np.random.rand(10)
    ak = np.random.rand()
    fun = lambda x: log_likelihood(x, sk, yk, ak)[0]
    dfun = lambda x: log_likelihood(x, sk, yk, ak)[1]
    return check_grad(fun, dfun, np.random.rand(100)) < 1e-4
    
assert check_likelihood()

#Initialize data
D = np.loadtxt('rosenbrock.dat', delimiter=",")
D = D[:500,:]
N = (D.shape[1] - 1)/2
positions = D[:,1:1+N]
s = positions[1:,:] - positions[:-1,:]
gradients = D[:,1+N:]
y = gradients[1:,:] - gradients[:-1,:]
alpha = D[1:,0]

#ADAM
theta = np.maximum(1e-4, np.ravel(.8*np.eye(N)))
theta = np.log(theta)
predicted, baseline = [], []

m = np.zeros(len(theta))
v = np.zeros(len(theta))
step_size=0.01
b1=0.9
b2=0.999
eps=10**-8
for i,(s_k, y_k, a_k) in enumerate(zip(s, y, alpha)):
    #Record
    W = np.exp(theta[:].reshape((N,N)))
    predicted += [dot(s_k, W.dot(y_k))/dot(y_k, y_k)]
    baseline += [dot(s_k, y_k)/dot(y_k, y_k)]
    #Update
    f, g = log_likelihood(theta, s_k, y_k, a_k)
    m = (1 - b1) * g      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(i + 1))    # Bias correction.
    vhat = v / (1 - b2**(i + 1))
    theta += step_size*mhat/(np.sqrt(vhat) + eps)
plt.plot(log(predicted), label = 'Predictive')
plt.plot(log(baseline), label = 'Heuristics')
plt.plot(log(alpha), label = 'Optimal')
plt.legend()
plt.show()
