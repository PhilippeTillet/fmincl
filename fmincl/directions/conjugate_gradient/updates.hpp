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
    virtual ~cg_update(){ }

    template<class BackendType>
    struct implementation : public implementation_base<BackendType>{
        implementation(detail::optimization_context<BackendType> & context) : implementation_base<BackendType>(context){ }
        virtual typename BackendType::ScalarType operator()(void) = 0;
    };
};

struct polak_ribiere : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        using implementation_base<BackendType>::context_;
        using typename implementation_base<BackendType>::ScalarType;
        using typename implementation_base<BackendType>::VectorType;
    public:
        implementation(polak_ribiere const &, detail::optimization_context<BackendType> & context) : cg_update::implementation<BackendType>(context) {
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
        std::size_t N_;
        VectorType tmp_;
    };
};


}

#endif
