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
#include <cmath>

#include "fmincl/utils.hpp"
#include "fmincl/tools/shared_ptr.hpp"
#include "fmincl/tools/typelist.hpp"
#include "fmincl/mapping.hpp"

namespace fmincl{

struct direction_tag{ virtual ~direction_tag(){ } };
template<class BackendType>
struct direction_implementation{
    virtual void operator()(detail::state<BackendType> & state) = 0;
};


/* =========================== *
 * CONJUGATE GRADIENTS
 * ===========================*/

//UPDATES
struct cg_update_tag{ virtual ~cg_update_tag(){ } };
template<class BackendType>
struct cg_update_implementation{
    virtual typename BackendType::ScalarType operator()(detail::state<BackendType> & state) = 0;
};

struct polak_ribiere_tag : public cg_update_tag{ };
template<class BackendType>
struct polak_ribiere_implementation : public cg_update_implementation<BackendType>{
    polak_ribiere_implementation(polak_ribiere_tag const &){ }
    typename BackendType::ScalarType operator()(detail::state<BackendType> & state){
        typename BackendType::VectorType tmp = state.g() - state.gm1();
        return BackendType::inner_prod(state.g(),tmp)/BackendType::inner_prod(state.gm1(),state.gm1());
    }
};

template<class BackendType>
struct cg_update_mapping{
    typedef implementation_from_tag<typename make_typelist<FMINCL_CREATE_MAPPING(polak_ribiere)>::type
                               ,cg_update_tag, cg_update_implementation<BackendType> > type;
};

//RESTARTS
struct cg_restart_tag{ virtual ~cg_restart_tag(){ } };
template<class BackendType>
struct cg_restart_implementation{
    virtual bool operator()(detail::state<BackendType> & state) = 0;
};

struct no_restart_tag : public cg_restart_tag{ };
template<class BackendType>
struct no_restart_implementation : public cg_restart_implementation<BackendType>{
    no_restart_implementation(no_restart_tag const & tag){ }
    bool operator()(detail::state<BackendType> & state) { return false; }
};

template<class BackendType>
struct cg_restart_mapping{
    typedef implementation_from_tag<typename make_typelist<FMINCL_CREATE_MAPPING(no_restart)>::type
                                   ,cg_restart_tag, cg_restart_implementation<BackendType> > type;
};

//CG
struct cg_tag : public direction_tag{
    cg_tag(cg_update_tag * _update = new polak_ribiere_tag(), cg_restart_tag * _restart = new no_restart_tag()) : update(_update), restart(_restart){ }
    tools::shared_ptr<cg_update_tag> update;
    tools::shared_ptr<cg_restart_tag> restart;
};
template<class BackendType>
class cg_implementation : public direction_implementation<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
public:
    cg_implementation(cg_tag const & cg_params) : update_implementation_(cg_update_mapping<BackendType>::type::create(*cg_params.update))
                                             ,restart_implementation_(cg_restart_mapping<BackendType>::type::create(*cg_params.restart)){ }
    void operator()(detail::state<BackendType> & state){
      if((*restart_implementation_)(state))
        state.p() = -state.g();
      else{
        ScalarType beta = (*update_implementation_)(state);
        state.p() = -state.g() + beta* state.p();
      }
    }
private:
    tools::shared_ptr<cg_update_implementation<BackendType> > update_implementation_;
    tools::shared_ptr<cg_restart_implementation<BackendType> > restart_implementation_;
};

/* =========================== *
 * QUASI NEWTON
 * ===========================*/

struct qn_update_tag{ virtual ~qn_update_tag(){ } };
template<class BackendType>
struct qn_update_implementation{
    virtual void operator()(detail::state<BackendType> & state) = 0;
};

struct lbfgs_tag : public qn_update_tag{
    lbfgs_tag(unsigned int _m = 4) : m(_m) { }
    unsigned int m;
};
template<class BackendType>
class lbfgs_implementation : public qn_update_implementation<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::MatrixType MatrixType;
public:
    lbfgs_implementation(lbfgs_tag const & _lbfgs) : vecs_(_lbfgs.m) { }

    void operator()(detail::state<BackendType> & state){
        unsigned int m = vecs_.size();
        for(unsigned int i = std::min(state.iter(),m)-1 ; i > 0  ; --i){
            vecs_[i] = vecs_[i-1];
        }
        vecs_[0].first = state.x() - state.xm1();
        vecs_[0].second = state.g() - state.gm1();

        std::vector<ScalarType> rhos(m);
        std::vector<ScalarType> alphas(m);

        VectorType q = BackendType::create_vector(state.dim());
        VectorType r = BackendType::create_vector(state.dim());

        int i = 0;
        q = state.g();
        for(; i < std::min(state.iter(),m) ; ++i){
            VectorType & s = vecs_[i].first;
            VectorType & y = vecs_[i].second;
            rhos[i] = 1.0d/BackendType::inner_prod(y,s);
            alphas[i] = rhos[i]*BackendType::inner_prod(s,q);
            q -= alphas[i]*y;
        }
        VectorType & sk = vecs_[0].first;
        VectorType & yk = vecs_[0].second;
        ScalarType scale = BackendType::inner_prod(sk,yk)/BackendType::inner_prod(yk,yk);
        r = scale*q;
        --i;
        for(; i >=0 ; --i){
            VectorType & s = vecs_[i].first;
            VectorType & y = vecs_[i].second;
            ScalarType beta = rhos[i]*BackendType::inner_prod(y,r);
            r += s*(alphas[i]-beta);
        }
        state.p() = -r;

        BackendType::delete_if_dynamically_allocated(q);
        BackendType::delete_if_dynamically_allocated(r);
    }
private:
    std::vector<std::pair<VectorType, VectorType> > vecs_;
};

struct bfgs_tag : public qn_update_tag{ };
template<class BackendType>
class bfgs_implementation : public qn_update_implementation<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::MatrixType MatrixType;
public:
    bfgs_implementation(bfgs_tag const &) : is_first_update_(true){ }

    void operator()(detail::state<BackendType> & state){
      VectorType s = state.x() - state.xm1();
      VectorType y = state.g() - state.gm1();

      VectorType Hy = BackendType::create_vector(state.dim());

      ScalarType ys = BackendType::inner_prod(s,y);
      if(is_first_update_==true){
        ScalarType yy = BackendType::inner_prod(y,y);
        ScalarType scale = ys/yy;
        BackendType::set_to_identity(Hk, state.dim());
        Hk *= scale;
        is_first_update_=false;
      }

      BackendType::prod(Hk,y,Hy);
      ScalarType yHy = BackendType::inner_prod(y,Hy);
      BackendType::rank_2_update(-1/ys,s,Hy,Hk);
      BackendType::rank_2_update(-1/ys,Hy,s,Hk);
      BackendType::rank_2_update(1/ys + yHy/pow(ys,2),s,s,Hk);

      BackendType::prod(Hk,state.g(),state.p());
      state.p() = -state.p();

      BackendType::delete_if_dynamically_allocated(Hy);
    }

private:
    MatrixType Hk;
    bool is_first_update_;
};

template<class BackendType>
struct qn_update_mapping{
    typedef implementation_from_tag<typename make_typelist<FMINCL_CREATE_MAPPING(lbfgs),FMINCL_CREATE_MAPPING(bfgs)>::type
                                   ,qn_update_tag, qn_update_implementation<BackendType> > type;
};

struct quasi_newton_tag : public direction_tag{
    quasi_newton_tag(qn_update_tag * _update = new lbfgs_tag()) : update(_update){ }
    tools::shared_ptr<qn_update_tag> update;
};
template<class BackendType>
class quasi_newton_implementation : public direction_implementation<BackendType>{
  public:
    quasi_newton_implementation(quasi_newton_tag const & tag) : update(qn_update_mapping<BackendType>::type::create(*tag.update)){ }
    virtual void operator()(detail::state<BackendType> & state){ (*update)(state);  }
private:
    tools::shared_ptr<qn_update_implementation<BackendType> > update;
};

template<class BackendType>
struct direction_mapping{
    typedef implementation_from_tag<typename make_typelist<FMINCL_CREATE_MAPPING(cg),FMINCL_CREATE_MAPPING(quasi_newton)>::type
                                   ,direction_tag, direction_implementation<BackendType> > type;
};




}

#endif
