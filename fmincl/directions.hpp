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

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

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
    viennacl::scalar<double> operator()(viennacl::vector<double> const & gk
                                        , viennacl::vector<double> const & gkm1){
        return viennacl::linalg::inner_prod(gk,  gk - gkm1)/viennacl::linalg::inner_prod(gkm1,gkm1);
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
        if(gkm1_.empty() || restart())
            state.p() = -state.g();
        else{
            viennacl::scalar<double> beta = compute_beta(state.g(), gkm1_);
            state.p() = -state.g() + beta* state.p();
        }
        gkm1_ = state.g();
    }

private:
    viennacl::vector<double> gkm1_;
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
        if(gkm1_.empty()){
            state.p() = -state.g();
        }
        else{
            viennacl::vector<double> skm1 = state.x() - xkm1_;
            viennacl::vector<double> ykm1 = state.g() - gkm1_;


            if(is_first_update_==true){
                viennacl::scalar<double> ipsy = viennacl::linalg::inner_prod(skm1,ykm1);
                viennacl::scalar<double> nykm1 = viennacl::linalg::inner_prod(ykm1,ykm1);
                viennacl::scalar<double> scale = ipsy/nykm1;
                Hk = viennacl::identity_matrix<double>(state.dim());
                Hk *= 1;
                is_first_update_=false;
            }

            viennacl::scalar<double> rho = (double)(1)/viennacl::linalg::inner_prod(skm1,ykm1);
            viennacl::scalar<double> rho2 = rho*rho;
            viennacl::vector<double> Hy = viennacl::linalg::prod(Hk,ykm1);
            viennacl::scalar<double> n2y = viennacl::linalg::inner_prod(ykm1,Hy);

            Hk -= rho*viennacl::linalg::outer_prod(Hy,skm1);
            Hk -= rho*viennacl::linalg::outer_prod(skm1,Hy);
            Hk += rho2*n2y*viennacl::linalg::outer_prod(skm1,skm1);
            Hk += rho*viennacl::linalg::outer_prod(skm1,skm1);


            viennacl::vector<double> tmp = viennacl::linalg::prod(Hk,state.g());
            state.p() = -tmp;
        }


        xkm1_ = state.x();
        gkm1_ = state.g();
    }

private:
    viennacl::vector<double> xkm1_;
    viennacl::vector<double> gkm1_;
    viennacl::matrix<double> Hk;
    bool is_first_update_;
};


}

#endif
