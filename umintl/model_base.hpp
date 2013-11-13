#ifndef UMINTL_EVALUATION_POLICY_HPP
#define UMINTL_EVALUATION_POLICY_HPP

#include <cstddef>
#include "umintl/forwards.h"
#include "umintl/optimization_context.hpp"
#include <cmath>

namespace umintl{

template<class BackendType>
struct model_base{
    virtual ~model_base(){ }
    virtual bool update(optimization_context<BackendType> & context) = 0;
    virtual value_gradient get_value_gradient_tag() const = 0;
    virtual hessian_vector_product get_hv_product_tag() const = 0;
};

template<class BackendType>
struct deterministic : public model_base<BackendType> {
    bool update(optimization_context<BackendType> &){ return false; }
    value_gradient get_value_gradient_tag() const { return value_gradient(DETERMINISTIC,0,0); }
    hessian_vector_product get_hv_product_tag() const { return hessian_vector_product(DETERMINISTIC,0,0); }
};

template<class BackendType>
struct mini_batch : public model_base<BackendType> {
  public:
    mini_batch(std::size_t sample_size, std::size_t dataset_size) : sample_size_(std::min(sample_size,dataset_size)), offset_(0), dataset_size_(dataset_size){ }
    bool update(optimization_context<BackendType> &){
      offset_=(offset_+sample_size_)%dataset_size_;
      return false;
    }
    value_gradient get_value_gradient_tag() const { return value_gradient(STOCHASTIC,dataset_size_,0); }
    hessian_vector_product get_hv_product_tag() const { return hessian_vector_product(STOCHASTIC,sample_size_,offset_); }
private:
    std::size_t sample_size_;
    std::size_t offset_;
    std::size_t dataset_size_;
};

template<class BackendType>
struct dynamically_sampled : public model_base<BackendType> {
  private:
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
  public:
    dynamically_sampled(double r, std::size_t initial_sample_size, std::size_t dataset_size, double theta = 0.5) : r_(r), S(initial_sample_size), offset_(0), N(dataset_size), theta_(theta){ }

    //      BackendType::set_to_value(var,0,c.N());
    //      for(std::size_t i = 0 ; i < S ; ++i){
    //        //tmp = (grad(xi) - grad(X)).^2
    //        //var += tmp
    //        c.fun().compute_value_gradient(c.x(),dummy,tmp,value_gradient(STOCHASTIC,1,offset_+i));
    //        for(std::size_t i = 0 ; i < c.N() ; ++i)
    //          var[i]+=std::pow(tmp[i]-c.g()[i],2);
    //      }
    //      BackendType::scale(c.N(),(ScalarType)1/(S-1),var);

    bool update(optimization_context<BackendType> & c){
      if(S==N)
        return false;
      else{
        VectorType var = BackendType::create_vector(c.N());
        c.fun().compute_gradient_variance(c.x(),c.g(),var,gradient_variance(STOCHASTIC,S,offset_));

        //is_descent_direction = norm1(var)/S*[(N-S)/(N-1)] <= theta^2*norm2(grad)^2
        ScalarType nrm1var = BackendType::asum(c.N(),var);
        ScalarType nrm2grad = BackendType::nrm2(c.N(),c.g());
        //std::gradient_variance << nrm1var*scal << " " << std::pow(theta_,2)*std::pow(nrm2grad,2) << std::endl;
        bool is_descent_direction = (nrm1var/S <= (std::pow(theta_,2)*std::pow(nrm2grad,2)));

        //Update parameters
        if(is_descent_direction==false){
          std::cout << "Augmenting sample size from " << S;
          S = nrm1var/std::pow(theta_*nrm2grad,2);
          S = std::min(S,N);
          std::cout << " to " << S << std::endl;
        }
        offset_=rand()%(N-S+1);
        BackendType::delete_if_dynamically_allocated(var);
        return true;
      }
    }

    value_gradient get_value_gradient_tag() const {
      return value_gradient(STOCHASTIC,S,offset_);
    }

    hessian_vector_product get_hv_product_tag() const {
      return hessian_vector_product(STOCHASTIC,r_*S,offset_);
    }
private:
    double theta_;
    double r_;
    std::size_t S;
    std::size_t offset_;
    std::size_t N;
};


}
#endif
