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
    virtual void operator()(void) = 0;
};


/* =========================== *
 * CONJUGATE GRADIENTS
 * ===========================*/

//UPDATES
struct cg_update_tag{ virtual ~cg_update_tag(){ } };
template<class BackendType>
struct cg_update_implementation{
    virtual typename BackendType::ScalarType operator()(void) = 0;
};

struct polak_ribiere_tag : public cg_update_tag{ };

template<class BackendType>
struct polak_ribiere_implementation : public cg_update_implementation<BackendType>{
private:
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
public:
    polak_ribiere_implementation(polak_ribiere_tag const &, detail::optimization_context<BackendType> & context) : context_(context){ }
    ScalarType operator()(){
        VectorType & g = context_.g();
        VectorType & gm1 = context_.gm1();
        unsigned int & N = context_.dim();

        VectorType tmp = g - gm1;
        return BackendType::dot(g,tmp)/BackendType::dot(gm1,gm1);
    }
private:
    detail::optimization_context<BackendType> & context_;
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
    virtual bool operator()(void) = 0;
};

struct no_restart_tag : public cg_restart_tag{ };
template<class BackendType>
struct no_restart_implementation : public cg_restart_implementation<BackendType>{
    no_restart_implementation(no_restart_tag const & tag, detail::optimization_context<BackendType> & context) { }
    bool operator()() { return false; }
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
    typedef typename BackendType::VectorType VectorType;
public:
    cg_implementation(cg_tag const & cg_params, detail::optimization_context<BackendType> & context) : context_(context)
                                                                                                      ,update_implementation_(cg_update_mapping<BackendType>::type::create(*cg_params.update, context))
                                                                                                      ,restart_implementation_(cg_restart_mapping<BackendType>::type::create(*cg_params.restart, context)){ }
    void operator()(){
      VectorType& p = context_.p();
      VectorType& g = context_.g();
      unsigned int& N = context_.dim();

      if((*restart_implementation_)())
        p = -g;
      else{
        ScalarType beta = (*update_implementation_)();
        p = -g + beta*p;
      }
    }
private:
    tools::shared_ptr<cg_update_implementation<BackendType> > update_implementation_;
    tools::shared_ptr<cg_restart_implementation<BackendType> > restart_implementation_;

    detail::optimization_context<BackendType> & context_;
};

/* =========================== *
 * QUASI NEWTON
 * ===========================*/

struct qn_update_tag{ virtual ~qn_update_tag(){ } };
template<class BackendType>
struct qn_update_implementation{
    virtual void operator()(void) = 0;
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
    lbfgs_implementation(lbfgs_tag const & _lbfgs, detail::optimization_context<BackendType> & context) : context_(context), vecs_(_lbfgs.m) { }

    void operator()(){
        VectorType & x=context_.x();
        VectorType & xm1=context_.xm1();
        VectorType & g=context_.g();
        VectorType & gm1=context_.gm1();
        VectorType & p=context_.p();
        unsigned int & iter = context_.iter();

        unsigned int m = vecs_.size();
        for(unsigned int i = std::min(iter,m)-1 ; i > 0  ; --i){
            vecs_[i] = vecs_[i-1];
        }
        vecs_[0].first = x - xm1;
        vecs_[0].second = g - gm1;

        std::vector<ScalarType> rhos(m);
        std::vector<ScalarType> alphas(m);

        VectorType q = BackendType::create_vector(context_.dim());
        VectorType r = BackendType::create_vector(context_.dim());

        int i = 0;
        q = g;
        for(; i < std::min(iter,m) ; ++i){
            VectorType & s = vecs_[i].first;
            VectorType & y = vecs_[i].second;
            rhos[i] = 1.0d/BackendType::dot(y,s);
            alphas[i] = rhos[i]*BackendType::dot(s,q);
            q -= alphas[i]*y;
        }
        VectorType & sk = vecs_[0].first;
        VectorType & yk = vecs_[0].second;
        ScalarType scale = BackendType::dot(sk,yk)/BackendType::dot(yk,yk);
        r = scale*q;
        --i;
        for(; i >=0 ; --i){
            VectorType & s = vecs_[i].first;
            VectorType & y = vecs_[i].second;
            ScalarType beta = rhos[i]*BackendType::dot(y,r);
            r += s*(alphas[i]-beta);
        }
        p = -r;

        BackendType::delete_if_dynamically_allocated(q);
        BackendType::delete_if_dynamically_allocated(r);
    }
private:
    std::vector<std::pair<VectorType, VectorType> > vecs_;
    detail::optimization_context<BackendType> & context_;
};

struct bfgs_tag : public qn_update_tag{ };
template<class BackendType>
class bfgs_implementation : public qn_update_implementation<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::MatrixType MatrixType;
public:
    bfgs_implementation(bfgs_tag const &, detail::optimization_context<BackendType> & context) : context_(context), is_first_update_(true){ }

    void operator()(){
      std::size_t N = context_.dim();

      VectorType Hy = BackendType::create_vector(N);
      VectorType s = BackendType::create_vector(N);
      VectorType y = BackendType::create_vector(N);

      //s = x - xm1
      BackendType::copy(context_.x(),s);
      BackendType::axpy(-1,context_.xm1(),s);

      //y = g - gm1
      BackendType::copy(context_.g(),y);
      BackendType::axpy(-1,context_.gm1(),y);

      ScalarType ys = BackendType::dot(s,y);
      if(is_first_update_==true){
        ScalarType yy = BackendType::dot(y,y);
        ScalarType scale = ys/yy;
        Hk = BackendType::create_matrix(N, N);
        BackendType::set_to_identity(Hk, N);
        BackendType::scale(scale,Hk);
        is_first_update_=false;
      }

      BackendType::gemv(Hk,y,Hy);
      ScalarType yHy = BackendType::dot(y,Hy);
      BackendType::syr2(-1/ys,s,Hy,Hk);
      BackendType::syr1(1/ys + yHy/pow(ys,2),s,Hk);

      BackendType::gemv(Hk,context_.g(),context_.p());
      BackendType::scale(-1,context_.p());

      BackendType::delete_if_dynamically_allocated(Hy);
    }

private:
    detail::optimization_context<BackendType> & context_;

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
    quasi_newton_implementation(quasi_newton_tag const & tag, detail::optimization_context<BackendType> & context) : update(qn_update_mapping<BackendType>::type::create(*tag.update, context)){ }
    virtual void operator()(){ (*update)();  }
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
