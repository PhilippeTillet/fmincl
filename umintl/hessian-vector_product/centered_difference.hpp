#ifndef HESSIAN_VECTOR_PRODUCT_CENTERED_DIFFERENCE_H
#define HESSIAN_VECTOR_PRODUCT_CENTERED_DIFFERENCE_H

#include "forwards.h"
#include "umintl/optimization_context.hpp"

namespace umintl{

    namespace hessian_vector_product{

    template<class BackendType>
    struct centered_difference : public base<BackendType>{
        using base<BackendType>::model_;
      private:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::MatrixType MatrixType;
      public:
       centered_difference(ScalarType _h = 1e-7) : h(_h){ }
       centered_difference(model_type_base * model, ScalarType _h = 1e-7) : base<BackendType>(model), h(_h){ }

       void init(optimization_context<BackendType> & c){
          c_ = &c;
          tmp_ = BackendType::create_vector(c.N());
          resleft_ = BackendType::create_vector(c.N());
        }
        void clean(optimization_context<BackendType> &){
          c_ = NULL;
          BackendType::delete_if_dynamically_allocated(tmp_);
          BackendType::delete_if_dynamically_allocated(resleft_);
        }
        void operator()(std::size_t N, VectorType const & b, VectorType & res)
        {
          ScalarType dummy;
          model_->update(c_->iter());

          //res = Grad(x+hb)
          BackendType::copy(N,c_->x(),tmp_); //tmp = x + hb
          BackendType::axpy(N,h,b,tmp_);
          c_->fun()(tmp_,dummy,res, value_gradient_tag(*model_));

          //resleft = Grad(x-hb)
          BackendType::copy(N,c_->x(),tmp_); //tmp = x - hb
          BackendType::axpy(N,-h,b,tmp_);
          c_->fun()(tmp_,dummy,resleft_, value_gradient_tag(*model_));

          //res-=resleft
          //res/=2h
          BackendType::axpy(N,-1,resleft_,res);
          BackendType::scale(N,1/(2*h),res);
        }
        ScalarType h;
      private:
        VectorType tmp_;
        VectorType resleft_;
        optimization_context<BackendType>  * c_;
    };

    }

}
#endif
