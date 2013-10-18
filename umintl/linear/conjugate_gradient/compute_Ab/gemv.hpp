/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINEAR_CONJUGATE_GRADIENT_COMPUTE_AB_GEMV_HPP_
#define UMINTL_LINEAR_CONJUGATE_GRADIENT_COMPUTE_AB_GEMV_HPP_

#include "forwards.h"
#include <cmath>

namespace umintl{

  namespace linear{

    namespace conjugate_gradient_detail{

      template<class BackendType>
      struct symv : public compute_Ab<BackendType>{
        private:
          typedef typename BackendType::MatrixType MatrixType;
          typedef typename BackendType::VectorType VectorType;
        public:
          symv(MatrixType const & A) : A_(A){ }
          void operator()(std::size_t N, VectorType const & b, VectorType & res)
          {
            BackendType::symv(N,1,A_,b,0,res);
          }
        private:
          MatrixType const & A_;
      };

    }

  }

}

#endif
