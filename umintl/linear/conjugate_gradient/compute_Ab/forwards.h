/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINEAR_CONJUGATE_GRADIENT_COMPUTE_AB_FORWARDS_H_
#define UMINTL_LINEAR_CONJUGATE_GRADIENT_COMPUTE_AB_FORWARDS_H_

#include <cstddef>

namespace umintl{

  namespace linear{

    namespace conjugate_gradient_detail{

      template<class BackendType>
      struct compute_Ab{
          virtual ~compute_Ab(){ }
          virtual void operator()(std::size_t N, typename BackendType::VectorType const & b, typename BackendType::VectorType & res) = 0;
      };

    }

  }

}



#endif
