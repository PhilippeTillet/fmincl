/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_UPDATE_BFGS_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_UPDATE_BFGS_HPP_

#include "forwards.h"
#include <vector>

namespace fmincl{

struct bfgs : public qn_update{
    template<class BackendType>
    class implementation : public qn_update::implementation<BackendType>{
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::MatrixType MatrixType;
    public:
        implementation(bfgs const &, detail::optimization_context<BackendType> & context) : reinitialize_(true){
            N_ = context.N();
            Hy_ = BackendType::create_vector(N_);
            s_ = BackendType::create_vector(N_);
            y_ = BackendType::create_vector(N_);
            H_ = BackendType::create_matrix(N_, N_);

            BackendType::set_to_value(Hy_,0,N_);
            BackendType::set_to_value(s_,0,N_);
            BackendType::set_to_value(y_,0,N_);

        }

        void erase_memory() {
            reinitialize_=true;
        }

        void operator()(detail::optimization_context<BackendType> & c){
          //s = x - xm1;
          BackendType::copy(N_,c.x(),s_);
          BackendType::axpy(N_,-1,c.xm1(),s_);

          //y = g - gm1;
          BackendType::copy(N_,c.g(),y_);
          BackendType::axpy(N_,-1,c.gm1(),y_);

          ScalarType ys = BackendType::dot(N_,s_,y_);

          if(reinitialize_)
            BackendType::set_to_diagonal(N_,H_,1);

          ScalarType gamma = 1;

          {
              BackendType::symv(N_,1,H_,y_,0,Hy_);
              ScalarType yHy = BackendType::dot(N_,y_,Hy_);
              ScalarType sg = BackendType::dot(N_,s_,c.gm1());
              ScalarType gHy = BackendType::dot(N_,c.gm1(),Hy_);
             if(ys/yHy>1)
                  gamma = ys/yHy;
              else if(sg/gHy<1)
                 gamma = sg/gHy;
              else
                  gamma = 1;
          }

          BackendType::scale(N_,N_,gamma,H_);
          BackendType::symv(N_,1,H_,y_,0,Hy_);
          ScalarType yHy = BackendType::dot(N_,y_,Hy_);

          //BFGS UPDATE
          //H_ += alpha*(s_*Hy' + Hy*s_') + beta*s_*s_';
          ScalarType alpha = -1/ys;
          ScalarType beta = 1/ys + yHy/pow(ys,2);
          BackendType::syr2(N_,alpha,s_,Hy_,H_);
          BackendType::syr1(N_,beta,s_,H_);

          //p = -H_*g
          BackendType::symv(N_,-1,H_,c.g(),0,c.p());

          if(reinitialize_)
              reinitialize_=false;
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

        bool reinitialize_;
    };
};


}

#endif
