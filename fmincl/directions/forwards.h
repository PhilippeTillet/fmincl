/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_FORWARDS_H
#define FMINCL_DIRECTIONS_FORWARDS_H

#include "fmincl/utils.hpp"

namespace fmincl{

struct direction{
    template<class BackendType>
    struct implementation{
        virtual void operator()(detail::optimization_context<BackendType> &) = 0;
        virtual double line_search_first_trial(detail::optimization_context<BackendType> & c){
            if(c.iter()==0)
                return std::min(static_cast<double>(1.0),1/BackendType::asum(c.N(),c.g()));
            else
                return std::min((double)1,2*(c.val() - c.valm1())/c.dphi_0());
        }
        virtual bool restart(detail::optimization_context<BackendType> &){ return false; }
        virtual ~implementation(){ }
    };
    virtual ~direction(){}
};


}

#endif
