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
    polak_ribiere_implementation(polak_ribiere_tag const &, detail::optimization_context<BackendType> & context) : context_(context){
        N_ = context_.dim();
        tmp_ = BackendType::create_vector(N_);
    }

    ScalarType operator()(){
        VectorType & g = context_.g();
        VectorType & gm1 = context_.gm1();

        //tmp_ = g - gm1;
        BackendType::copy(N_,g, tmp_);
        BackendType::axpy(N_,-1,gm1,tmp_);
        return BackendType::dot(N_,g,tmp_)/BackendType::dot(N_,gm1,gm1);
    }

    ~polak_ribiere_implementation(){
        BackendType::delete_if_dynamically_allocated(tmp_);
    }
private:
    detail::optimization_context<BackendType> & context_;

    std::size_t N_;
    VectorType tmp_;
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
      VectorType const & g = context_.g();
      VectorType & p = context_.p();
      std::size_t N = context_.dim();

      if((*restart_implementation_)()){

        //p = -g;
        BackendType::copy(N,g,p);
        BackendType::scale(N,-1,p);
      }
      else{
        ScalarType beta = (*update_implementation_)();

        //p = -g + beta*p;
        BackendType::scale(N,beta,p);
        BackendType::axpy(N,-1,g,p);
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

    struct storage_pair{
        VectorType s;
        VectorType y;
    };

    VectorType & s(std::size_t i) { return vecs_[i].s; }
    VectorType & y(std::size_t i) { return vecs_[i].y; }

public:
    lbfgs_implementation(lbfgs_tag const & tag, detail::optimization_context<BackendType> & context) : tag_(tag), context_(context), vecs_(tag.m){
        N_ = context_.dim();

        q_ = BackendType::create_vector(N_);
        r_ = BackendType::create_vector(N_);

        for(unsigned int i = 0 ; i < tag_.m ; ++i){
            vecs_[i].s = BackendType::create_vector(N_);
            vecs_[i].y = BackendType::create_vector(N_);
        }
    }

    void operator()(){
        //Initizalization of aliases
        VectorType const & x=context_.x();
        VectorType const & xm1=context_.xm1();
        VectorType const & g=context_.g();
        VectorType const & gm1=context_.gm1();
        VectorType & p=context_.p();

        unsigned int const & iter = context_.iter();
        unsigned int const & m = tag_.m;

        std::vector<ScalarType> rhos(m);
        std::vector<ScalarType> alphas(m);

        //Algorithm


        //Updates storage
        for(unsigned int i = std::min(iter,m)-1 ; i > 0  ; --i){
            BackendType::copy(N_,s(i-1), s(i));
            BackendType::copy(N_,y(i-1), y(i));
        }

        //s(0) = x - xm1;
        BackendType::copy(N_,x,s(0));
        BackendType::axpy(N_,-1,xm1,s(0));

        //y(0) = g - gm1;
        BackendType::copy(N_,g,y(0));
        BackendType::axpy(N_,-1,gm1,y(0));


        BackendType::copy(N_,g,q_);
        int i = 0;
        for(; i < std::min(iter,m) ; ++i){
            rhos[i] = 1.0d/BackendType::dot(N_,y(i),s(i));
            alphas[i] = rhos[i]*BackendType::dot(N_,s(i),q_);
            //q_ = q - alphas[i]*y(i);
            BackendType::axpy(N_,-alphas[i],y(i),q_);
        }
        ScalarType scale = BackendType::dot(N_,s(0),y(0))/BackendType::dot(N_,y(0),y(0));

        //r_ = scale*q_;
        BackendType::copy(N_,q_,r_);
        BackendType::scale(N_,scale,r_);

        --i;
        for(; i >=0 ; --i){
            ScalarType beta = rhos[i]*BackendType::dot(N_,y(i),r_);
            //r_ = r_ + (alphas[i]-beta)*s(i)
            BackendType::axpy(N_,alphas[i]-beta,s(i),r_);
        }

        //p = -r_;
        BackendType::copy(N_,r_,p);
        BackendType::scale(N_,-1,p);
    }

    ~lbfgs_implementation(){
        BackendType::delete_if_dynamically_allocated(q_);
        BackendType::delete_if_dynamically_allocated(r_);

        for(unsigned int i = 0 ; i < tag_.m ; ++i){
            BackendType::delete_if_dynamically_allocated(s(i));
            BackendType::delete_if_dynamically_allocated(y(i));
        }
    }

private:
    lbfgs_tag const & tag_;

    std::size_t N_;

    VectorType q_;
    VectorType r_;

    std::vector<storage_pair> vecs_;
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
      //Aliases initialization
      VectorType & x = context_.x();
      VectorType & xm1 = context_.xm1();
      VectorType & g = context_.g();
      VectorType & gm1 = context_.gm1();
      VectorType & p = context_.p();

      //Algorithm

      //s = x - xm1;
      BackendType::copy(N_,x,s_);
      BackendType::axpy(N_,-1,xm1,s_);

      //y = g - gm1;
      BackendType::copy(N_,g,y_);
      BackendType::axpy(N_,-1,gm1,y_);

      ScalarType ys = BackendType::dot(N_,s_,y_);
      if(is_first_update_==true){
        BackendType::set_to_identity(N_,H_);

        ScalarType yy = BackendType::dot(N_,y_,y_);
        ScalarType scale = ys/yy;
        BackendType::scale(N_,N_,scale,H_);
        is_first_update_=false;
      }

      //Hy_ = H*y
      BackendType::symv(N_,1,H_,y_,0,Hy_);
      ScalarType yHy = BackendType::dot(N_,y_,Hy_);

      //H_ += (-1/ys)*(s_*Hy' + Hy*s_')
      BackendType::syr2(N_,-1/ys,s_,Hy_,H_);

      //H_ += (1/ys + yHy/pow(ys,2))*s_*s_'
      BackendType::syr1(N_,1/ys + yHy/pow(ys,2),s_,H_);

      //p = -H_*g
      BackendType::symv(N_,1,H_,g,0,p);
      BackendType::scale(N_,-1,p);
    }

    ~bfgs_implementation(){
        BackendType::delete_if_dynamically_allocated(Hy_);
        BackendType::delete_if_dynamically_allocated(s_);
        BackendType::delete_if_dynamically_allocated(y_);

        BackendType::delete_if_dynamically_allocated(H_);
    }

private:
    detail::optimization_context<BackendType> & context_;

    std::size_t N_;

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
