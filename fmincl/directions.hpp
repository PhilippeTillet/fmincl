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
      unsigned int const & N = context_.dim();

      VectorType const & g = context_.g();
      VectorType & p = context_.p();

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
    lbfgs_implementation(lbfgs_tag const & tag, detail::optimization_context<BackendType> & context) : tag_(tag), context_(context), vecs_(tag.m){
        N_ = context_.dim();

        q_ = BackendType::create_vector(N_);
        r_ = BackendType::create_vector(N_);
    }

    void operator()(){
        //Initizalization of aliases
        VectorType const & x=context_.x();
        VectorType const & xm1=context_.xm1();
        VectorType const & g=context_.g();
        VectorType const & gm1=context_.gm1();
        VectorType & p=context_.p();
        unsigned int iter = context_.iter();
        unsigned int m = tag_.m;

        //Algorithm
        for(unsigned int i = std::min(iter,m)-1 ; i > 0  ; --i){
            vecs_[i] = vecs_[i-1];
        }
        vecs_[0].first = x - xm1;
        vecs_[0].second = g - gm1;

        std::vector<ScalarType> rhos(m);
        std::vector<ScalarType> alphas(m);


        int i = 0;
        q_ = g;
        for(; i < std::min(iter,m) ; ++i){
            VectorType & s = vecs_[i].first;
            VectorType & y = vecs_[i].second;
            rhos[i] = 1.0d/BackendType::dot(y,s);
            alphas[i] = rhos[i]*BackendType::dot(s,q_);
            q_ -= alphas[i]*y;
        }
        VectorType & sk = vecs_[0].first;
        VectorType & yk = vecs_[0].second;
        ScalarType scale = BackendType::dot(sk,yk)/BackendType::dot(yk,yk);
        r_ = scale*q_;
        --i;
        for(; i >=0 ; --i){
            VectorType & s = vecs_[i].first;
            VectorType & y = vecs_[i].second;
            ScalarType beta = rhos[i]*BackendType::dot(y,r_);
            r_ += s*(alphas[i]-beta);
        }
        p = -r_;
    }

    ~lbfgs_implementation(){
        BackendType::delete_if_dynamically_allocated(q_);
        BackendType::delete_if_dynamically_allocated(r_);
    }

private:
    lbfgs_tag const & tag_;

    unsigned int N_;

    VectorType q_;
    VectorType r_;

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
    bfgs_implementation(bfgs_tag const &, detail::optimization_context<BackendType> & context) : context_(context), is_first_update_(true){
        N_ = context_.dim();
        Hy_ = BackendType::create_vector(N_);
        s_ = BackendType::create_vector(N_);
        y_ = BackendType::create_vector(N_);
        H_ = BackendType::create_matrix(N_, N_);
    }

    void operator()(){
      VectorType & x = context_.x();
      VectorType & xm1 = context_.xm1();
      VectorType & g = context_.g();
      VectorType & gm1 = context_.gm1();
      VectorType & p = context_.p();

      //s = x - xm1
      BackendType::copy(x,s_);
      BackendType::axpy(-1,xm1,s_);

      //y = g - gm1
      BackendType::copy(g,y_);
      BackendType::axpy(-1,gm1,y_);

      ScalarType ys = BackendType::dot(s_,y_);
      if(is_first_update_==true){
        BackendType::set_to_identity(H_, N_);

        ScalarType yy = BackendType::dot(y_,y_);
        ScalarType scale = ys/yy;
        BackendType::scale(scale,H_);
        is_first_update_=false;
      }

      BackendType::gemv(H_,y_,Hy_);
      ScalarType yHy = BackendType::dot(y_,Hy_);
      BackendType::syr2(-1/ys,s_,Hy_,H_);
      BackendType::syr1(1/ys + yHy/pow(ys,2),s_,H_);

      BackendType::gemv(H_,g,p);
      BackendType::scale(-1,p);
    }

    ~bfgs_implementation(){
        BackendType::delete_if_dynamically_allocated(Hy_);
        BackendType::delete_if_dynamically_allocated(s_);
        BackendType::delete_if_dynamically_allocated(y_);

        BackendType::delete_if_dynamically_allocated(H_);
    }

private:
    detail::optimization_context<BackendType> & context_;

    unsigned int N_;

    VectorType Hy_;
    VectorType s_;
    VectorType y_;

    MatrixType H_;

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
