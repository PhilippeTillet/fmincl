#ifndef HESSIAN_VECTOR_PRODUCT_PROVIDED_FUNCTION_H
#define HESSIAN_VECTOR_PRODUCT_PROVIDED_FUNCTION_H

#include "forwards.h"
#include "umintl/optimization_context.hpp"

namespace umintl{

    namespace hessian_vector_product{

        template<class BackendType>
        struct provided_function : public base<BackendType>{
            using base<BackendType>::model_;
          private:
            typedef typename BackendType::ScalarType ScalarType;
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::MatrixType MatrixType;
          public:
            provided_function(model_type_base * model) : base<BackendType>(model){ }
            void init(optimization_context<BackendType> & c){ c_ = &c; }
            void clean(optimization_context<BackendType> &){ c_ = NULL; }
            void operator()(std::size_t N, VectorType const & v, VectorType & Hv){
                model_->update(c_->iter());
                c_->fun()(c_->x(), v, Hv, hessian_vector_product_tag(*model_));
            }
          private:
            optimization_context<BackendType>  * c_;
        };

    }

}
#endif
