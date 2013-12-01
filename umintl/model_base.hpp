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
    dynamically_sampled(double r, std::size_t S0, std::size_t dataset_size, double theta = 0.5) : r_(r), S(std::min(S0,dataset_size)), offset_(0), H_offset_(0), N(dataset_size), theta_(theta){ }

    bool update(optimization_context<BackendType> & c){
//      {
//        VectorType var = BackendType::create_vector(c.N());
//        VectorType tmp = BackendType::create_vector(c.N());
//        VectorType Hv = BackendType::create_vector(c.N());
//        BackendType::set_to_value(var,0,c.N());
//        c.fun().compute_hv_product(c.x(),c.g(),c.g(),Hv,hessian_vector_product(STOCHASTIC,S,offset_));

//        for(std::size_t i = 0 ; i < S ; ++i){
//          //tmp = (grad(xi) - grad(X)).^2
//          //var += tmp
//          c.fun().compute_hv_product(c.x(),c.g(),c.g(),tmp,hessian_vector_product(STOCHASTIC,1,offset_+i));
//          for(std::size_t i = 0 ; i < c.N() ; ++i)
//            var[i]+=std::pow(tmp[i]-Hv[i],2);
//        }
//        BackendType::scale(c.N(),(ScalarType)1/(S-1),var);
//        for(std::size_t i = 0 ; i < c.N() ; ++i)
//          std::cout << var[i] << " ";
//        std::cout << std::endl;

//        c.fun().compute_hv_product_variance(c.x(),c.g(), var, hv_product_variance(STOCHASTIC,S,offset_));

//        for(std::size_t i = 0 ; i < c.N() ; ++i)
//          std::cout << var[i] << " ";
//        std::cout << std::endl;

//        BackendType::delete_if_dynamically_allocated(var);
//        BackendType::delete_if_dynamically_allocated(tmp);
//        BackendType::delete_if_dynamically_allocated(Hv);
//      }


      if(S==N){
        H_offset_=(H_offset_+(int)(r_*S))%(S - (int)(r_*S) + 1);
        return false;
      }
      else{
        VectorType var = BackendType::create_vector(c.N());
        c.fun().compute_gradient_variance(c.x(),var,gradient_variance(STOCHASTIC,S,offset_));

        //is_descent_direction = norm1(var)/S*[(N-S)/(N-1)] <= theta^2*norm2(grad)^2
        ScalarType nrm1var = BackendType::asum(c.N(),var);
        ScalarType nrm2grad = BackendType::nrm2(c.N(),c.g());
        //std::gradient_variance << nrm1var*scal << " " << std::pow(theta_,2)*std::pow(nrm2grad,2) << std::endl;
        bool is_descent_direction = (nrm1var/S <= (std::pow(theta_,2)*std::pow(nrm2grad,2)));

        //Update parameters
        std::size_t old_S = S;
        if(is_descent_direction==false){
          S = nrm1var/std::pow(theta_*nrm2grad,2);
          S = std::min(S,N);
          if(S>N/2)
            S=N;
          std::cout << "Augmenting sample size from " << old_S << " to " << S << std::endl;
        }
        offset_=rand()%(N-S+1);

        if(is_descent_direction==false)
          H_offset_ = 0;
        else
          H_offset_=(rand())%(S - (int)(r_*S) + 1);

        BackendType::delete_if_dynamically_allocated(var);
        return true;
      }
    }

    value_gradient get_value_gradient_tag() const {
      return value_gradient(STOCHASTIC,S,offset_);
    }

    hessian_vector_product get_hv_product_tag() const {
      return hessian_vector_product(STOCHASTIC,r_*S,H_offset_+offset_);
    }
private:
    double theta_;
    double r_;
    std::size_t S;
    std::size_t offset_;
    std::size_t H_offset_;
    std::size_t N;
};


}
#endif
