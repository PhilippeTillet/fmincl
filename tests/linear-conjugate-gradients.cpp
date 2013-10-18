/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include <iostream>

#include "umintl/backends/cblas.hpp"
#include "umintl/linear/conjugate_gradient.hpp"


typedef float ScalarType;
typedef umintl::backend::cblas_types<ScalarType> BackendType;

int main(){
  std::size_t N = 100;
  ScalarType * b = new ScalarType[N];
  ScalarType * x = new ScalarType[N];
  ScalarType * x0 = new ScalarType[N];
  ScalarType * sqrtA = new ScalarType[N*N];
  ScalarType * A = new ScalarType[N*N];

  for(std::size_t i = 0 ; i < N ; ++i)
    x0[i] = 0;
  for(std::size_t i = 0 ; i < N ; ++i)
    b[i] = 1;
  for(std::size_t i = 0 ; i < N*N ; ++i)
    sqrtA[i] = 0.001*(ScalarType)rand()/RAND_MAX;
  for(std::size_t i = 0 ; i < N*N ; ++i)
    A[i] = 0;

  for(std::size_t i = 0 ; i < N ; ++i)
    for(std::size_t j = 0 ; j < N ; ++j)
      for(std::size_t k = 0 ; k < N ; ++k)
        A[i*N+j]+=sqrtA[i*N+k]*sqrtA[j*N+k];

  ScalarType epsilon = 1e-8;
  umintl::linear::conjugate_gradient<BackendType> conjugate_gradient(100,epsilon);
  umintl::linear::conjugate_gradient<BackendType>::return_code ret = conjugate_gradient(N,A,x0,b,x);

  if(ret==umintl::linear::conjugate_gradient<BackendType>::SUCCESS)
    return EXIT_SUCCESS;

  return EXIT_FAILURE;

}
