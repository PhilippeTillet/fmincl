/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_BACKEND_HPP
#define FMINCL_BACKEND_HPP

#if not defined FMINCL_WITH_VIENNACL\
    && not defined FMINCL_WITH_EIGEN
#error "Please specify a backend"
#endif

#ifdef FMINCL_WITH_VIENNACL
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#endif

#ifdef FMINCL_WITH_EIGEN
#include "Eigen/Dense"
#endif

namespace fmincl{

  namespace backend{

#ifdef FMINCL_WITH_VIENNACL
    typedef viennacl::vector<double> VECTOR_TYPE;
    typedef viennacl::matrix<double, viennacl::row_major> MATRIX_TYPE;
    typedef double SCALAR_TYPE;
    static SCALAR_TYPE inner_prod(VECTOR_TYPE const & x, VECTOR_TYPE const & y){
      return viennacl::linalg::inner_prod(x,y);
    }
    static void set_to_identity(MATRIX_TYPE & M, unsigned int n){
      M = viennacl::identity_matrix<SCALAR_TYPE>(n);
    }
    static void prod(MATRIX_TYPE const& M, VECTOR_TYPE const & x, VECTOR_TYPE & res){
      res = viennacl::linalg::prod(M,x);
    }
    static void rank_2_update(SCALAR_TYPE const & alpha, VECTOR_TYPE const & x, VECTOR_TYPE const & y, MATRIX_TYPE & res){
      res+=alpha*viennacl::linalg::outer_prod(x,y);
    }
#endif

#ifdef FMINCL_WITH_EIGEN
    typedef VectorXd VECTOR_TYPE;
    typedef MatrixXd MATRIX_TYPE
    typedef double SCALAR_TYPE
    static SCALAR_TYPE inner_prod(VECTOR_TYPE const & x, VECTOR_TYPE const & y){
      return x.dot(y);
    }
    static void set_to_identity(MATRIX_TYPE & M, unsigned int n){
      M = MATRIX_TYPE::Identity(n, n);
    }
    static void prod(MATRIX_TYPE const& M, VECTOR_TYPE const & x, VECTOR_TYPE & res){
      res = M*x;
    }
    static void rank_2_update(SCALAR_TYPE const & alpha, VECTOR_TYPE const & x, VECTOR_TYPE const & y, MATRIX_TYPE & res){
      res+=alpha*x*y.transpose();
    }
#endif
  }

}
#endif
