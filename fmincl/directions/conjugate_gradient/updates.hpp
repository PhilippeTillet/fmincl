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

#include <cmath>

namespace fmincl{

struct cg_update{
    virtual ~cg_update(){ }

    template<class BackendType>
    struct implementation : public implementation_base<BackendType>{
        implementation(detail::optimization_context<BackendType> & context) : implementation_base<BackendType>(context){ }
        virtual double operator()(void) = 0;
    };
};

struct polak_ribiere : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        using implementation_base<BackendType>::context_;
        using typename implementation_base<BackendType>::VectorType;
    public:
        implementation(polak_ribiere const &, detail::optimization_context<BackendType> & context) : cg_update::implementation<BackendType>(context), g_(context_.g()), gm1_(context_.gm1()){
            N_ = context_.dim();
            tmp_ = BackendType::create_vector(N_);
        }

        ~implementation(){  BackendType::delete_if_dynamically_allocated(tmp_); }

        double operator()(){
            //tmp_ = g - gm1;
            BackendType::copy(N_,g_, tmp_);
            BackendType::axpy(N_,-1,gm1_,tmp_);
            return std::max(BackendType::dot(N_,g_,tmp_)/BackendType::dot(N_,gm1_,gm1_),(double)0);
        }
    private:
        std::size_t N_;
        VectorType & g_;
        VectorType & gm1_;
        VectorType tmp_;
    };
};


struct fletcher_reeves : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        using implementation_base<BackendType>::context_;
        using typename implementation_base<BackendType>::VectorType;
    public:
        implementation(fletcher_reeves const &, detail::optimization_context<BackendType> & context) : cg_update::implementation<BackendType>(context), g_(context.g()), gm1_(context.gm1()){
            N_ = context_.dim();
        }
        double operator()(){
            return BackendType::dot(N_,g_,g_)/BackendType::dot(N_,gm1_,gm1_);
        }
    private:
        std::size_t N_;
        VectorType & g_;
        VectorType & gm1_;
    };
};



}

#endif
