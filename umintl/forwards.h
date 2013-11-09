#ifndef UMINTL_FORWARDS_H
#define UMINTL_FORWARDS_H

namespace umintl{


template<class BackendType>
class optimization_context;

class model_type_base;

struct tag_base {
    tag_base(model_type_base const & _model) : model(_model){ }
    model_type_base const & model;
};

struct value_tag : public tag_base { value_tag(model_type_base const & model) : tag_base(model){} };
struct gradient_tag : public tag_base { gradient_tag(model_type_base const & model) : tag_base(model){} };
struct value_gradient_tag : public tag_base { value_gradient_tag(model_type_base const & model) : tag_base(model){} };
struct hessian_vector_product_tag : public tag_base { hessian_vector_product_tag(model_type_base const & model) : tag_base(model){} };

}
#endif
