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
    struct implementation{
        virtual double operator()(detail::optimization_context<BackendType> &) = 0;
        virtual ~implementation(){ }
    };
};

struct polak_ribiere : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(polak_ribiere const &, detail::optimization_context<BackendType> & context){
            N_ = context.dim();
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


struct fletcher_reeves : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(fletcher_reeves const &, detail::optimization_context<BackendType> & context){
            N_ = context.dim();
        }
        double operator()(detail::optimization_context<BackendType> & c){
            return BackendType::dot(N_,c.g(),c.g())/BackendType::dot(N_,c.gm1(),c.gm1());
        }
    private:
        std::size_t N_;
    };
};


struct gilbert_nocedal : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(gilbert_nocedal const &, detail::optimization_context<BackendType> &){ }
        double operator()(detail::optimization_context<BackendType> & context){
            double betaPR = polak_ribiere::implementation<BackendType>(polak_ribiere(),context)(context);
            double betaFR = fletcher_reeves::implementation<BackendType>(fletcher_reeves(),context)(context);
            return std::min(betaPR,betaFR);
        }
    };
};


}

#endif
