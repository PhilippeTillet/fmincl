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

#include <vector>

#include "fmincl/backend.hpp"
#include "fmincl/utils.hpp"
#include "fmincl/tools/shared_ptr.hpp"

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

//UPDATES
class cg_update{
  public:
    virtual backend::SCALAR_TYPE operator()(backend::VECTOR_TYPE const & gk, backend::VECTOR_TYPE const & gkm1) = 0;
};

class polak_ribiere : public cg_update{
  public:
    backend::SCALAR_TYPE operator()(backend::VECTOR_TYPE const & gk, backend::VECTOR_TYPE const & gkm1){
        return backend::inner_prod(gk,  gk - gkm1)/backend::inner_prod(gkm1,gkm1);
    }
};

//RESTARTS
class cg_restart{
  public:
    virtual bool operator()(detail::state & state) = 0;
};


class no_restart : public cg_restart{
  public:
    bool operator()(detail::state & state) {
      return false;
    }
};


//CG
class cg : public detail::direction_base{
public:
    cg(cg_update * update = new polak_ribiere(), cg_restart * restart = new no_restart()) : update_(update), restart_(restart){

    }

    void operator()(detail::state & state){
      if((*restart_)(state))
        state.p() = -state.g();
      else{
        backend::SCALAR_TYPE beta = (*update_)(state.g(), state.gm1());
        state.p() = -state.g() + beta* state.p();
      }
    }

private:
    tools::shared_ptr<cg_update> update_;
    tools::shared_ptr<cg_restart> restart_;
};


/* =========================== *
 * QUASI NEWTON
 * ===========================*/

class qn_update{
  public:
    virtual void operator()(detail::state & state) = 0;
};

class lbfgs : public qn_update{
  public:
    lbfgs(unsigned int m = 8) : m_(m), vecs_(m_) { }

    void operator()(detail::state & state){
      for(unsigned int i = std::min(state.iter(),m_)-1 ; i > 0  ; --i){
        vecs_[i] = vecs_[i-1];
      }
      vecs_[0].first = state.x() - state.xm1();
      vecs_[0].second = state.g() - state.gm1();

      std::vector<double> rhos(m_);
      std::vector<double> alphas(m_);

      int i = 0;
      backend::VECTOR_TYPE q = state.g();
      for(; i < std::min(state.iter(),m_) ; ++i){
        backend::VECTOR_TYPE & s = vecs_[i].first;
        backend::VECTOR_TYPE & y = vecs_[i].second;
        rhos[i] = 1.0d/backend::inner_prod(y,s);
        alphas[i] = rhos[i]*backend::inner_prod(s,q);
        q -= alphas[i]*y;
      }
      backend::VECTOR_TYPE & sk = vecs_[0].first;
      backend::VECTOR_TYPE & yk = vecs_[0].second;
      double scale = backend::inner_prod(sk,yk)/backend::inner_prod(yk,yk);
      backend::VECTOR_TYPE r = scale*q;
      --i;
      for(; i >=0 ; --i){
        backend::VECTOR_TYPE & s = vecs_[i].first;
        backend::VECTOR_TYPE & y = vecs_[i].second;
        double beta = rhos[i]*backend::inner_prod(y,r);
        r += s*(alphas[i]-beta);
      }
      state.p() = -r;
    }

  private:
    unsigned int m_;
    std::vector<std::pair<backend::VECTOR_TYPE, backend::VECTOR_TYPE> > vecs_;
};

class bfgs : public qn_update{
public:
    bfgs() : is_first_update_(true){ }

    void operator()(detail::state & state){
      backend::VECTOR_TYPE s = state.x() - state.xm1();
      backend::VECTOR_TYPE y = state.g() - state.gm1();
      if(is_first_update_==true){
        backend::SCALAR_TYPE ipsy = backend::inner_prod(s,y);
        backend::SCALAR_TYPE nykm1 = backend::inner_prod(y,y);
        backend::SCALAR_TYPE scale = ipsy/nykm1;
        backend::set_to_identity(Hk, state.dim());
        Hk *= scale;
        is_first_update_=false;
      }

      double ys = backend::inner_prod(s,y);
      backend::VECTOR_TYPE Hy(backend::size1(Hk));
      backend::prod(Hk,y,Hy);
      double yHy = backend::inner_prod(y,Hy);
      double gamma = ys/yHy;
      backend::VECTOR_TYPE v(backend::size1(Hk));
      v = std::sqrt(yHy)*(s/ys - Hy/yHy);
      Hk = gamma*Hk;
      backend::rank_2_update(-gamma/yHy,Hy,Hy,Hk);
      backend::rank_2_update(gamma,v,v,Hk);
      backend::rank_2_update(1/ys,s,s,Hk);

      backend::VECTOR_TYPE tmp(backend::size1(Hk));
      backend::prod(Hk,state.g(),tmp);
      state.p() = -tmp;
    }

private:
    backend::MATRIX_TYPE Hk;
    bool is_first_update_;
};

class quasi_newton : public detail::direction_base{
  public:
    quasi_newton(qn_update * update = new lbfgs(8)) : update_(update){ }

    virtual void operator()(detail::state & state){
      (*update_)(state);
    }

  private:
    tools::shared_ptr<qn_update> update_;
};





}

#endif
