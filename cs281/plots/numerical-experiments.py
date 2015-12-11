import numpy as np
import matplotlib.pyplot as plt

for benchmark in ['mnist-test', 'mnist-train', 'rosenbrock']:
    standard = np.loadtxt(benchmark + '/standard.dat',delimiter=',')
    rescaled = np.loadtxt(benchmark + '/rescaled.dat',delimiter=',')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(standard[:,0], standard[:,1], linewidth=2, label = 'Standard BFGS', color = 'blue')
    ax.plot(rescaled[:,0], rescaled[:,1], linewidth=2, label = 'Our algorithm', color = 'red')
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Test error (%)' if benchmark=='mnist-test' else 'Objective value')
    if benchmark != 'mnist-test':
        ax.set_yscale('log', nonposy='clip')
    if benchmark == 'mnist-train':
        ax.set_ylim([2e4, 2e5])
    ax.legend(loc = 'lower left' if benchmark == 'rosenbrock' else 'upper right')
    plt.savefig(benchmark + '.png')
