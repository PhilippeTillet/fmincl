/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_LINE_SEARCH_FORWARDS_H_
#define FMINCL_LINE_SEARCH_FORWARDS_H_

#include <cmath>

#include "fmincl/directions/forwards.h"
#include "fmincl/directions/quasi_newton.hpp"

#include "fmincl/forwards.h"
#include "fmincl/utils.hpp"
#include "fmincl/mapping.hpp"


namespace fmincl{

  template<class ScalarType>
  inline ScalarType cubicmin(ScalarType a,ScalarType b, ScalarType fa, ScalarType fb, ScalarType dfa, ScalarType dfb, ScalarType xmin, ScalarType xmax){
    ScalarType d1 = dfa + dfb - 3*(fa - fb)/(a-b);
    ScalarType delta = pow(d1,2) - dfa*dfb;
    if(delta<0)
      return (xmin+xmax)/2;
    ScalarType d2 = std::sqrt(delta);
    ScalarType x = b - (b - a)*((dfb + d2 - d1)/(dfb - dfa + 2*d2));
    if(isnan(x))
      return (xmin+xmax)/2;
    return std::min(std::max(x,xmin),xmax);
  }

  template<class ScalarType>
  inline ScalarType cubicmin(ScalarType a,ScalarType b, ScalarType fa, ScalarType fb, ScalarType dfa, ScalarType dfb){
    return cubicmin(a,b,fa,fb,dfa,dfb,std::min(a,b), std::max(a,b));
  }


  template<class BackendType>
  struct line_search_result{
    private:
      typedef typename BackendType::VectorType VectorType;
      typedef typename BackendType::ScalarType ScalarType;

      //NonCopyable, we do not want useless temporaries here
      line_search_result(line_search_result const &){ }
      line_search_result & operator=(line_search_result const &){ }
    public:
      line_search_result(std::size_t dim) : has_failed(false), best_x(BackendType::create_vector(dim)), best_g(BackendType::create_vector(dim)){ }
      ~line_search_result() {
          BackendType::delete_if_dynamically_allocated(best_x);
          BackendType::delete_if_dynamically_allocated(best_g);
      }
      bool has_failed;
      ScalarType best_phi;
      VectorType best_x;
      VectorType best_g;
  };

  struct line_search{
      template<class BackendType>
      struct implementation{
          typedef typename BackendType::ScalarType ScalarType;
      public:
          virtual void operator()(line_search_result<BackendType> & res,fmincl::direction::implementation<BackendType> * direction, detail::optimization_context<BackendType> & context, ScalarType ai) = 0;
          virtual ~implementation(){ }
      };


      virtual ~line_search(){ }
  };


}

#endif
