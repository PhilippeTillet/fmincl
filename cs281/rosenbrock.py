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
    #Unroll
    W = np.reshape(theta[:], (s_k.size,s_k.size))
    #Predicted mean
    mu_k = dot(s_k, dot(W, y_k))/dot(y_k, y_k)
    #Likelihood
    x = .5*(a_k - mu_k)**2/(mu_k**2*a_k)
    f = .5*log(_lambda) - _lambda*x
    #Gradient
    dtheta =  _lambda*(a_k - mu_k)/(a_k*dot(y_k,y_k)) * (1/mu_k**2 + (a_k - mu_k)/(mu_k**3)) * np.ravel(outer(s_k, y_k))
    return f, dtheta

def check_likelihood():
    sk = np.random.rand(10)
    yk = np.random.rand(10)
    ak = np.random.rand()
    fun = lambda x: log_likelihood(x, sk, yk, ak)[0]
    dfun = lambda x: log_likelihood(x, sk, yk, ak)[1]
    return check_grad(fun, dfun, np.random.rand(100)) < 1e-4

def adam(grad, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    return x
    
assert check_likelihood()
D = np.loadtxt('rosenbrock.dat', delimiter=",")
D = D[:500,:]
#Dimensionality of the parameters space
N = (D.shape[1] - 1)/2
#Iterates of the optimization
positions = D[:,1:1+N]
s = positions[1:,:] - positions[:-1,:]
#Gradient of the iterates
gradients = D[:,1+N:]
y = gradients[1:,:] - gradients[:-1,:]
#Build features
alpha = D[1:,0]
#ADAM
theta = np.ravel(10*np.eye(N))
predicted, baseline = [], []

m = np.zeros(len(theta))
v = np.zeros(len(theta))
step_size=0.001
b1=0.9
b2=0.999
eps=10**-8
for i,(s_k, y_k, a_k) in enumerate(zip(s, y, alpha)):
    #Record
    W = theta[:].reshape((N,N))
    predicted += [dot(s_k, W.dot(y_k))/dot(y_k, y_k)]
    baseline += [dot(s_k, y_k)/dot(y_k, y_k)]
    #Update
    f, g = log_likelihood(theta, s_k, y_k, a_k)
    m = (1 - b1) * g      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(i + 1))    # Bias correction.
    vhat = v / (1 - b2**(i + 1))
    theta += step_size*mhat/(np.sqrt(vhat) + eps)
    print np.diag(W)
plt.plot(log(predicted), label = 'Predictive')
plt.plot(log(baseline), label = 'Heuristics')
plt.plot(log(alpha), label = 'Optimal')
plt.legend()
plt.show()
