#ifndef HESSIAN_VECTOR_PRODUCT_FORWARDS_H
#define HESSIAN_VECTOR_PRODUCT_FORWARDS_H

#include "umintl/linear/conjugate_gradient/compute_Ab/forwards.h"
#include "umintl/model_type/forwards.h"
#include "umintl/model_type/deterministic.hpp"

namespace umintl{

    namespace hessian_vector_product{

        template<class BackendType>
        struct base : public linear::conjugate_gradient_detail::compute_Ab<BackendType>{
            base(model_type_base * model = new model_type::deterministic()) : model_(model){ }
            virtual void init(optimization_context<BackendType> &){ }
            virtual void clean(optimization_context<BackendType> &){ }
        protected:
            tools::shared_ptr<model_type_base> model_;
        };

    }

}
#endif
