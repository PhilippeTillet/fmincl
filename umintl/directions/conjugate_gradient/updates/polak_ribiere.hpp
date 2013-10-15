/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_CONJUGATE_GRADIENT_UPDATES_POLAK_RIBIERE_HPP_
#define UMINTL_CONJUGATE_GRADIENT_UPDATES_POLAK_RIBIERE_HPP_

#include "forwards.h"
#include <cmath>

namespace umintl{

template<class BackendType>
struct polak_ribiere : public cg_update<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;

    void init(detail::optimization_context<BackendType> & c){
        tmp_ = BackendType::create_vector(c.N());
    }

    void clean(detail::optimization_context<BackendType> &){
        BackendType::delete_if_dynamically_allocated(tmp_);
    }

    ScalarType operator()(detail::optimization_context<BackendType> & c){
        //tmp_ = g - gm1;
        BackendType::copy(c.N(),c.g(), tmp_);
        BackendType::axpy(c.N(),-1,c.gm1(),tmp_);
        return std::max(BackendType::dot(c.N(),c.g(),tmp_)/BackendType::dot(c.N(),c.gm1(),c.gm1()),(ScalarType)0);
    }
private:
    VectorType tmp_;
};

}

#endif
