UMinTL
======

UMinTL is a C++ library for gradient-base nonlinear optimization (BFGS, Newton-Raphson, Conjugate Gradient, Newton-CG, L-BFGS). 

It is flexible and allows for different compute backends to be used:
* cBLAS - use cBLAS for internal computations
* Eigen - use Eigen for internal computations

The interface is user-friendly and several examples can be found.

BFGS is re-scaled using information obtained from the line-search procedure, as prescribed by theoretically sound MAP estimation of the latent true Hessian matrix.

Examples
=======

We provide various examples:
* algorithms.cpp: Shows how to use various algorithms to optimize the same function using UMinTL -- with possibly custom stopping criterion. [Require cBLAS]
* rosenbrock.cpp: Shows the capabilities of the rescaled BFGS on the Rosenbrock function. [Require cBLAS]
* mnist.cpp: Shows the capabilities of the rescaled BFGS on a MultiLayer Perceptron for the MNIST dataset. [Require Eigen]
