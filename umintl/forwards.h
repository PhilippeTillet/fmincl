#ifndef UMINTL_FORWARDS_H
#define UMINTL_FORWARDS_H

#include <cstddef>
#include "umintl/tools/shared_ptr.hpp"

namespace umintl{


template<class BackendType>
class optimization_context;

enum model_type_tag {  DETERMINISTIC, STOCHASTIC };

struct operation_tag {
    operation_tag(model_type_tag const & _model, std::size_t _sample_size, std::size_t _offset) : model(_model), sample_size(_sample_size), offset(_offset){ }
    model_type_tag model;
    std::size_t sample_size;
    std::size_t offset;
};

struct value_gradient : public operation_tag { value_gradient(model_type_tag const & _model, std::size_t _sample_size, std::size_t _offset) : operation_tag(_model,_sample_size,_offset){ } };
struct hessian_vector_product : public operation_tag { hessian_vector_product(model_type_tag const & _model, std::size_t _sample_size, std::size_t _offset) : operation_tag(_model,_sample_size,_offset){ } };

enum computation_type{ CENTERED_DIFFERENCE, FORWARD_DIFFERENCE, PROVIDED };

struct model_type_base{
    virtual ~model_type_base(){ }
    virtual void update(std::size_t i) = 0;
    virtual value_gradient get_value_gradient_tag() const = 0;
    virtual hessian_vector_product get_hv_product_tag() const = 0;
};

struct deterministic : public model_type_base{
    void update(std::size_t){ }
    value_gradient get_value_gradient_tag() const { return value_gradient(DETERMINISTIC,0,0); }
    hessian_vector_product get_hv_product_tag() const { return hessian_vector_product(DETERMINISTIC,0,0); }
};

struct semi_stochastic : public model_type_base{
  public:
    semi_stochastic(std::size_t sample_size, std::size_t dataset_size) : sample_size_(sample_size), offset_(0), dataset_size_(dataset_size){ }
    void update(std::size_t i){ offset_= ((i*sample_size_) % dataset_size_); }
    value_gradient get_value_gradient_tag() const { return value_gradient(STOCHASTIC,dataset_size_,0); }
    hessian_vector_product get_hv_product_tag() const { return hessian_vector_product(STOCHASTIC,std::min(sample_size_, dataset_size_-offset_),offset_); }
private:
    std::size_t sample_size_;
    std::size_t offset_;
    std::size_t dataset_size_;
};

struct evaluation_policy_type{
    evaluation_policy_type() : hessian_vector_product_computation(CENTERED_DIFFERENCE), model(new deterministic()){ }
    evaluation_policy_type(computation_type _hessian_vector_product_computation, tools::shared_ptr<model_type_base> _model) : hessian_vector_product_computation(_hessian_vector_product_computation), model(_model){ }
    computation_type hessian_vector_product_computation;
    tools::shared_ptr<model_type_base> model;
};

}
#endif
