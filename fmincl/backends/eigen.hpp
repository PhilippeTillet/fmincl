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

        static VectorType create_vector(std::size_t n)
        { return VectorType(n); }
        static MatrixType create_matrix(std::size_t m, std::size_t n)
        { return MatrixType(m,n); }
        static void delete_if_dynamically_allocated(VectorType const &) { }

        static void copy(VectorType const & from, VectorType & to)
        { to = from; }
        static void axpy(ScalarType alpha, VectorType const & x, VectorType & y)
        {  y = alpha*x + y; }
        static void scale(ScalarType alpha, VectorType & x)
        { x = alpha*x; }
        static void scale(ScalarType alpha, MatrixType & A)
        { A = alpha*A; }
        static ScalarType asum(VectorType const & x)
        { return x.array().abs().sum(); }
        static ScalarType nrm2(VectorType const & x)
        { return x.norm(); }
        static ScalarType dot(VectorType const & x, VectorType const & y)
        { return x.dot(y); }
        static void gemv(MatrixType const& M, VectorType const & x, VectorType & res)
        { res = M*x;  }
        static void syr1(ScalarType const & alpha, VectorType const & x, MatrixType & res)
        { res+=alpha*x*x.transpose(); }
        static void syr2(ScalarType const & alpha, VectorType const & x, VectorType const & y, MatrixType & res)
        { res+=alpha*x*y.transpose() + Eigen::internal::conj(alpha)*y*x.transpose(); }
        static void set_to_identity(MatrixType & M, unsigned int n)
        { M = MatrixType::Identity(n, n); }

        static size_t size1(MatrixType const & M){ return M.rows(); }
        static size_t size2(MatrixType const & M){ return M.cols(); }
        static size_t size(VectorType const & v){ return v.size(); }
        static bool is_empty(VectorType const & v){ return size(v)==0; }
    };








  }

}

#endif
