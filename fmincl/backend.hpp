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


#include "Eigen/Dense"

namespace fmincl{

  namespace backend{

    template<class _ScalarType>
    struct EigenTypes{
        typedef _ScalarType ScalarType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorType;
        typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> MatrixType;
    };

    template<class EigenType>
    struct scalartype_of{
        typedef typename EigenType::Scalar type;
    };

    template<class VectorType>
    inline typename scalartype_of<VectorType>::type abs_sum(Eigen::MatrixBase<VectorType> const & x){
      return x.array().abs().sum();
    }

    template<class VectorType1, class VectorType2>
    inline typename scalartype_of<VectorType1>::type inner_prod(Eigen::MatrixBase<VectorType1> const & x, Eigen::MatrixBase<VectorType2> const & y){
      return x.dot(y);
    }

    template<class MatrixType>
    inline void set_to_identity(Eigen::MatrixBase<MatrixType> & M, unsigned int n){
      M = MatrixType::Identity(n, n);
    }

    template<class MatrixType, class VectorType1, class VectorType2>
    inline void prod(Eigen::MatrixBase<MatrixType> const& M, Eigen::MatrixBase<VectorType1> const & x, Eigen::MatrixBase<VectorType2> & res){
      res = M*x;
    }

    template<class ScalarType, class MatrixType, class VectorType>
    inline void rank_2_update(ScalarType const & alpha, Eigen::MatrixBase<VectorType> const & x, Eigen::MatrixBase<VectorType> const & y, Eigen::MatrixBase<MatrixType> & res){
      res+=alpha*x*y.transpose();
    }

    template<class MatrixType>
    inline size_t size1(Eigen::MatrixBase<MatrixType> const & M){
      return M.rows();
    }

    template<class MatrixType>
    inline size_t size2(Eigen::MatrixBase<MatrixType> const & M){
      return M.cols();
    }

    template<class VectorType>
    inline size_t size(Eigen::MatrixBase<VectorType> const & v){
      return v.size();
    }

    template<class VectorType>
    inline bool is_empty(Eigen::MatrixBase<VectorType> const & v){
      return size(v)==0;
    }

  }

}

#endif
