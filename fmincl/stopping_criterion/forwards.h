/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_STOPPING_CRITERION_FORWARDS_H
#define FMINCL_STOPPING_CRITERION_FORWARDS_H


namespace fmincl{

struct stopping_criterion{
    virtual ~stopping_criterion(){ }

    template<class BackendType>
    struct implementation{
        virtual bool operator()() = 0;
        virtual ~implementation(){ }
    };
};



}

#endif
