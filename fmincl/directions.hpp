/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_HPP_
#define FMINCL_DIRECTIONS_HPP_

#include "fmincl/backend.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

namespace detail{

class direction_base{
public:
    virtual void operator()(detail::state & state) = 0;
};

}

/* =========================== *
 * CONJUGATE GRADIENTS
 * ===========================*/

struct polak_ribiere{
    backend::SCALAR_TYPE operator()(backend::VECTOR_TYPE const & gk
                                        , backend::VECTOR_TYPE const & gkm1){
        return backend::inner_prod(gk,  gk - gkm1)/backend::inner_prod(gkm1,gkm1);
    }
};

struct no_restart{
    bool operator()(){
        return false;
    }
};


template<class BETA_POLICY = polak_ribiere, class RESTART_POLICY = no_restart>
class cg : public detail::direction_base{
public:
    cg() { }
    void operator()(detail::state & state){
        if(backend::is_empty(gkm1_) || restart())
            state.p() = -state.g();
        else{
            backend::SCALAR_TYPE beta = compute_beta(state.g(), gkm1_);
            state.p() = -state.g() + beta* state.p();
        }
        gkm1_ = state.g();
    }

private:
    backend::VECTOR_TYPE gkm1_;
    BETA_POLICY compute_beta;
    RESTART_POLICY restart;
};


/* =========================== *
 * QUASI NEWTON
 * ===========================*/


class bfgs{

};

template<class UPDATE>
class quasi_newton : public detail::direction_base{
public:
    quasi_newton() : is_first_update_(true) {

    }

    void operator()(detail::state & state){
        if(backend::is_empty(gkm1_)){
            state.p() = -state.g();
        }
        else{
            backend::VECTOR_TYPE skm1 = state.x() - xkm1_;
            backend::VECTOR_TYPE ykm1 = state.g() - gkm1_;


            if(is_first_update_==true){
                backend::SCALAR_TYPE ipsy = backend::inner_prod(skm1,ykm1);
                backend::SCALAR_TYPE nykm1 = backend::inner_prod(ykm1,ykm1);
                backend::SCALAR_TYPE scale = ipsy/nykm1;
                backend::set_to_identity(Hk, state.dim());
                Hk *= scale;
                is_first_update_=false;
            }

            backend::SCALAR_TYPE rho = (double)(1)/backend::inner_prod(skm1,ykm1);
            backend::SCALAR_TYPE rho2 = rho*rho;
            backend::VECTOR_TYPE Hy(backend::size1(Hk));
            backend::prod(Hk,ykm1,Hy);
            backend::SCALAR_TYPE n2y = backend::inner_prod(ykm1,Hy);

            backend::rank_2_update(-rho,Hy,skm1,Hk);
            backend::rank_2_update(-rho,skm1,Hy,Hk);
            backend::rank_2_update(rho2*n2y,skm1,skm1,Hk);
            backend::rank_2_update(rho,skm1,skm1,Hk);

            backend::VECTOR_TYPE tmp(backend::size1(Hk));
            backend::prod(Hk,state.g(),tmp);

            state.p() = -tmp;
        }
        xkm1_ = state.x();
        gkm1_ = state.g();
    }

private:
    backend::VECTOR_TYPE xkm1_;
    backend::VECTOR_TYPE gkm1_;
    backend::MATRIX_TYPE Hk;
    bool is_first_update_;
};


}

#endif
