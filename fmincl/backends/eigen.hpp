/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_BACKENDS_EIGEN_HPP
#define FMINCL_BACKENDS_EIGEN_HPP


#include "Eigen/Dense"

namespace fmincl{

  namespace backend{

    template<class _ScalarType>
    struct EigenTypes{
        typedef _ScalarType ScalarType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

        static ScalarType norm_1(VectorType const & x){ return x.array().abs().sum();  }
        static ScalarType inner_prod(VectorType const & x, VectorType const & y){ return x.dot(y); }
        static void set_to_identity(MatrixType & M, unsigned int n){ M = MatrixType::Identity(n, n); }
        static void prod(MatrixType const& M, VectorType const & x, VectorType & res){ res = M*x; }
        static void rank_2_update(ScalarType const & alpha, VectorType const & x, VectorType const & y, MatrixType & res){ res+=alpha*x*y.transpose(); }
        static size_t size1(MatrixType const & M){ return M.rows(); }
        static size_t size2(MatrixType const & M){ return M.cols(); }
        static size_t size(VectorType const & v){ return v.size(); }
        static bool is_empty(VectorType const & v){ return size(v)==0; }
    };








  }

}

#endif
