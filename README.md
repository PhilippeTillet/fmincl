UMinTL
======

This project is a generic unconstrained minimization library relying heavily on template metaprogramming.

##Features

* Use **any** Linear Algebra backend

The linear algebra routines are well separated from the algorithm implementations.
You can pass a custom backend in the optimization procedure. This backend should just include a few typedefs, linear algebra routine and procedures to allocate/delete a Vector/Matrix.
For now, cblas and ViennaCL are available, for using either CPU or OpenCL in the optimization procedure and the objective function.

* An **extendable** library

UMinTL relies on template. You may therefore write your own stopping criterion (eg a cross-validation error if you're doing machine learning). More experienced user may also write their own restarting condition for the conjugate gradient, or their own update direction.

* A **clear** interface

UMinTL provides a clear API, where any functor can be passed to the optimization algorithm. The cumbersomeness of the templates is softened using type-erasure mechanism.

* A **robust** package

UMinTL was extensively tested on the test suite described in "Testing Unconstrained Optimization Software" (JJ Moré and al.). However, it will never be robust enough. If the procedure fails on your particular problem, please *report*.

* **Multiple procedures** available

The library supports BFGS, L-BFGS, and Nonlinear Conjugate Gradient. Several updates and restart procedures are available for the conjugate gradient method.
The line-search is done using the strong wolfe-powell conditions.
