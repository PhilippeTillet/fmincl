/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_UPDATE_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_UPDATE_HPP_

#include "fmincl/mapping.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

struct qn_update{
    template<class BackendType>
    struct implementation : public implementation_base<BackendType>{
        implementation(detail::optimization_context<BackendType> & context) : implementation_base<BackendType>(context){ }
        virtual void operator()(void) = 0;
    };

    virtual ~qn_update(){ }
};

struct lbfgs : public qn_update{
    lbfgs(unsigned int _m = 4) : m(_m) { }
    unsigned int m;

    template<class BackendType>
    class implementation : public qn_update::implementation<BackendType>{
        using implementation_base<BackendType>::context_;
        using typename implementation_base<BackendType>::VectorType;
        using typename implementation_base<BackendType>::MatrixType;

        struct storage_pair{
            VectorType s;
            VectorType y;
        };

        VectorType & s(std::size_t i) { return vecs_[i].s; }
        VectorType & y(std::size_t i) { return vecs_[i].y; }

    public:
        implementation(lbfgs const & tag, detail::optimization_context<BackendType> & context) : qn_update::implementation<BackendType>(context), tag_(tag), vecs_(tag.m){
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

            std::vector<double> rhos(m);
            std::vector<double> alphas(m);

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
            for(; i < (int)std::min(iter,m) ; ++i){
                rhos[i] = static_cast<double>(1)/BackendType::dot(N_,y(i),s(i));
                alphas[i] = rhos[i]*BackendType::dot(N_,s(i),q_);
                //q_ = q - alphas[i]*y(i);
                BackendType::axpy(N_,-alphas[i],y(i),q_);
            }
            double scale = BackendType::dot(N_,s(0),y(0))/BackendType::dot(N_,y(0),y(0));

            //r_ = scale*q_;
            BackendType::copy(N_,q_,r_);
            BackendType::scale(N_,scale,r_);

            --i;
            for(; i >=0 ; --i){
                double beta = rhos[i]*BackendType::dot(N_,y(i),r_);
                //r_ = r_ + (alphas[i]-beta)*s(i)
                BackendType::axpy(N_,alphas[i]-beta,s(i),r_);
            }

            //p = -r_;
            BackendType::copy(N_,r_,p);
            BackendType::scale(N_,-1,p);
        }

        ~implementation(){
            BackendType::delete_if_dynamically_allocated(q_);
            BackendType::delete_if_dynamically_allocated(r_);

            for(unsigned int i = 0 ; i < tag_.m ; ++i){
                BackendType::delete_if_dynamically_allocated(s(i));
                BackendType::delete_if_dynamically_allocated(y(i));
            }
        }

    private:
        lbfgs const & tag_;
        std::size_t N_;
        VectorType q_;
        VectorType r_;
        std::vector<storage_pair> vecs_;
    };
};



struct bfgs : public qn_update{
    template<class BackendType>
    class implementation : public qn_update::implementation<BackendType>{
        using implementation_base<BackendType>::context_;
        using typename implementation_base<BackendType>::VectorType;
        using typename implementation_base<BackendType>::MatrixType;
    public:
        implementation(bfgs const &, detail::optimization_context<BackendType> & context) : qn_update::implementation<BackendType>(context), is_first_update_(true){
            N_ = context_.dim();
            Hy_ = BackendType::create_vector(N_);
            s_ = BackendType::create_vector(N_);
            y_ = BackendType::create_vector(N_);
            H_ = BackendType::create_matrix(N_, N_);

            BackendType::set_to_value(Hy_,0,N_);
            BackendType::set_to_value(s_,0,N_);
            BackendType::set_to_value(y_,0,N_);

        }

        void operator()(){
          //s = x - xm1;
          BackendType::copy(N_,context_.x(),s_);
          BackendType::axpy(N_,-1,context_.xm1(),s_);

          //y = g - gm1;
          BackendType::copy(N_,context_.g(),y_);
          BackendType::axpy(N_,-1,context_.gm1(),y_);

          double ys = BackendType::dot(N_,s_,y_);

          if(is_first_update_)
            BackendType::set_to_diagonal(N_,H_,1);

          double gamma = 1;

          {
              BackendType::symv(N_,1,H_,y_,0,Hy_);
              double yHy = BackendType::dot(N_,y_,Hy_);
              double sg = BackendType::dot(N_,s_,context_.gm1());
              double gHy = BackendType::dot(N_,context_.gm1(),Hy_);
             if(ys/yHy>1)
                  gamma = ys/yHy;
              else if(sg/gHy<1)
                 gamma = sg/gHy;
              else
                  gamma = 1;
          }

          BackendType::scale(N_,N_,gamma,H_);
          BackendType::symv(N_,1,H_,y_,0,Hy_);
          double yHy = BackendType::dot(N_,y_,Hy_);

          //BFGS UPDATE
          //H_ += alpha*(s_*Hy' + Hy*s_') + beta*s_*s_';
          double alpha = -1/ys;
          double beta = 1/ys + yHy/pow(ys,2);
          BackendType::syr2(N_,alpha,s_,Hy_,H_);
          BackendType::syr1(N_,beta,s_,H_);

          //p = -H_*g
          BackendType::symv(N_,-1,H_,context_.g(),0,context_.p());

          if(is_first_update_) is_first_update_=false;
        }

        ~implementation(){
            BackendType::delete_if_dynamically_allocated(Hy_);
            BackendType::delete_if_dynamically_allocated(s_);
            BackendType::delete_if_dynamically_allocated(y_);

            BackendType::delete_if_dynamically_allocated(H_);
        }

    private:
        std::size_t N_;

        VectorType Hy_;
        VectorType s_;
        VectorType y_;

        MatrixType H_;

        bool is_first_update_;
    };
};


}

#endif
