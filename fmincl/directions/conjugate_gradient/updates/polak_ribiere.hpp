/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CONJUGATE_GRADIENT_UPDATES_POLAK_RIBIERE_HPP_
#define FMINCL_CONJUGATE_GRADIENT_UPDATES_POLAK_RIBIERE_HPP_

#include "forwards.h"
#include <cmath>

namespace fmincl{


struct polak_ribiere : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(polak_ribiere const &, detail::optimization_context<BackendType> & context){
            N_ = context.N();
            tmp_ = BackendType::create_vector(N_);
        }

        ~implementation(){  BackendType::delete_if_dynamically_allocated(tmp_); }

        double operator()(detail::optimization_context<BackendType> & c){
            //tmp_ = g - gm1;
            BackendType::copy(N_,c.g(), tmp_);
            BackendType::axpy(N_,-1,c.gm1(),tmp_);
            return std::max(BackendType::dot(N_,c.g(),tmp_)/BackendType::dot(N_,c.gm1(),c.gm1()),(double)0);
        }
    private:
        std::size_t N_;
        VectorType tmp_;
    };
};

}

#endif
