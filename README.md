UMinTL
======

This project is a generic unconstrained minimization library relying heavily on template metaprogramming.

##Features [ UMinTL 1.0 ]

* Compatible with **any** Linear Algebra backend

The linear algebra routines are well separated from the algorithm implementations.
You can plug your own linear algebra backend into the optimization procedure. This backend should just include a few typedefs, linear algebra routine and procedures to allocate/delete a Vector/Matrix.
For now, backends for BLAS (tested on mwblas provided by matlab), cBLAS (tested on OpenBlas) and Eigen are supported. CuBlas or ViennaCL backends should also work out-of-the box, but are not provided, yet.

* **Lightweight**, **portable** and **headers-only**

UMinTL is C++03-compliant and does not require any external package.

* An **extendable** library

UMinTL relies on C++ templates. You may therefore write your own stopping criterion (eg a cross-validation error if you're doing machine learning). More experienced user may also write their own restarting condition for the conjugate gradient, or their own update direction.

* A **clear** interface

UMinTL provides a clear C++ API, relying on functors rather than functions..

* A **robust** package

UMinTL was extensively tested on the test suite described in "Testing Unconstrained Optimization Software" (JJ Moré and al.). However, I am aware that it will never be robust enough. If the procedure fails on your particular problem, please *report*.

* **Efficient* optimization routines

The library supports BFGS, L-BFGS, and Nonlinear Conjugate Gradient (Several updates and restart procedures).
The line-search is done using the strong wolfe-powell conditions.
The implementations are inspired from the renowed minfunc package for MATLAB (www.di.ens.fr/~mschmidt/Software/minFunc.html‎)


## Incoming Features [ UMinTL 1.1 ]

* Hessian-free optimization

Hessian-free optimization has shown encouraging results in the Machine Learning litterature. It is a fundamental building block of the unconstrained optimization framework, which should absolutely be provided.

* Mini-batch / Stochastic and Semi-Stochastic procedures

For now, UMinTL only provides batch methods. Mini-Batch/Stochastic/Semi-stochastic approaches should also be provided.

* Supported GPU backend(s)

At least ViennaCL backend should be provided, in order to carry out the optimization process on heterogeneous devices, or to avoid CPU-to-GPU transfers (when optimizing on the CPU and evaluating the objective function on the GPU)
