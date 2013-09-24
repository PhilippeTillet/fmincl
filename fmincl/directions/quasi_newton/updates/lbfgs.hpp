/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_UPDATE_LBFGS_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_UPDATE_LBFGS_HPP_

#include "forwards.h"
#include <vector>

namespace fmincl{

struct lbfgs : public qn_update{
    lbfgs(unsigned int _m = 4) : m(_m) { }
    unsigned int m;

    template<class BackendType>
    class implementation : public qn_update::implementation<BackendType>{
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
        implementation(lbfgs const & tag, detail::optimization_context<BackendType> & context) : tag_(tag), vecs_(tag.m){
            N_ = context.N();

            q_ = BackendType::create_vector(N_);
            r_ = BackendType::create_vector(N_);

            for(unsigned int i = 0 ; i < tag_.m ; ++i){
                vecs_[i].s = BackendType::create_vector(N_);
                vecs_[i].y = BackendType::create_vector(N_);
            }

            n_valid_pairs_ = 0;
        }

        void erase_memory() {
            n_valid_pairs_ = 0;
        }

        void operator()(detail::optimization_context<BackendType> & c){
            std::vector<ScalarType> rhos(tag_.m);
            std::vector<ScalarType> alphas(tag_.m);

            //Algorithm
            n_valid_pairs_ = std::min(n_valid_pairs_+1,tag_.m);

            //Updates storage
            for(unsigned int i = n_valid_pairs_-1 ; i > 0  ; --i){
                BackendType::copy(N_,s(i-1), s(i));
                BackendType::copy(N_,y(i-1), y(i));
            }

            //s(0) = x - xm1;
            BackendType::copy(N_,c.x(),s(0));
            BackendType::axpy(N_,-1,c.xm1(),s(0));

            //y(0) = g - gm1;
            BackendType::copy(N_,c.g(),y(0));
            BackendType::axpy(N_,-1,c.gm1(),y(0));


            BackendType::copy(N_,c.g(),q_);
            int i = 0;
            for(; i < (int)n_valid_pairs_ ; ++i){
                rhos[i] = static_cast<ScalarType>(1)/BackendType::dot(N_,y(i),s(i));
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
            BackendType::copy(N_,r_,c.p());
            BackendType::scale(N_,-1,c.p());
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
        unsigned int n_valid_pairs_;
    };
};


}

#endif
