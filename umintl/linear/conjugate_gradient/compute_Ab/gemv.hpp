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
      struct gemv : public compute_Ab<BackendType>{
          void operator()(std::size_t M, std::size_t N, typename BackendType::MatrixType const & A, typename BackendType::VectorType const & b, typename BackendType::VectorType & res){
            BackendType::gemv(M,N,1,A,b,0,res);
          }
      };

    }

  }

}

#endif
