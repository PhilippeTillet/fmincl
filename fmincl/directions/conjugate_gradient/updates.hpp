/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CONJUGATE_GRADIENT_UPDATES_HPP_
#define FMINCL_CONJUGATE_GRADIENT_UPDATES_HPP_

#include "fmincl/utils.hpp"

namespace fmincl{

struct cg_update{
    template<class BackendType>
    struct implementation{
        virtual typename BackendType::ScalarType operator()(void) = 0;
        virtual ~implementation(){ }
    };
    virtual ~cg_update(){ }
};

struct polak_ribiere : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(polak_ribiere const &, detail::optimization_context<BackendType> & context) : context_(context){
            N_ = context_.dim();
            tmp_ = BackendType::create_vector(N_);
        }

        ScalarType operator()(){
            VectorType & g = context_.g();
            VectorType & gm1 = context_.gm1();

            //tmp_ = g - gm1;
            BackendType::copy(N_,g, tmp_);
            BackendType::axpy(N_,-1,gm1,tmp_);
            return BackendType::dot(N_,g,tmp_)/BackendType::dot(N_,gm1,gm1);
        }

        ~implementation(){
            BackendType::delete_if_dynamically_allocated(tmp_);
        }
    private:
        detail::optimization_context<BackendType> & context_;
        std::size_t N_;
        VectorType tmp_;
    };
};


}

#endif
