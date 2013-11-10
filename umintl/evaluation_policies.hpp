/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_EVALUATION_POLICIES_HPP_
#define UMINTL_EVALUATION_POLICIES_HPP_

#include "umintl/tools/shared_ptr.hpp"

#include "umintl/model_type/forwards.h"
#include "umintl/model_type/deterministic.hpp"

#include "umintl/forwards.h"

namespace umintl{

    struct hv_product_evaluation_policy{
        enum computation_type{ CENTERED_DIFFERENCE, FORWARD_DIFFERENCE, PROVIDED };
        hv_product_evaluation_policy() : computation(CENTERED_DIFFERENCE), model(new model_type::deterministic){ }
        computation_type computation;
        tools::shared_ptr<model_type_base> model;
    };

    struct gradient_evaluation_policy{
        gradient_evaluation_policy() : model(new model_type::deterministic()){ }
        tools::shared_ptr<model_type_base> model;
    };

    struct value_evaluation_policy{
        value_evaluation_policy() : model(new model_type::deterministic()){ }
        tools::shared_ptr<model_type_base> model;
    };

    struct value_gradient_evaluation_policy{
        value_gradient_evaluation_policy() : model(new model_type::deterministic()){ }
        tools::shared_ptr<model_type_base> model;
    };


    struct evaluation_policies_type{
      hv_product_evaluation_policy hv_product;
      gradient_evaluation_policy gradient;
      value_evaluation_policy value;
      value_gradient_evaluation_policy value_gradient;
    };

}


#endif
