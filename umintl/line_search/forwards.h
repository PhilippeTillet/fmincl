/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINE_SEARCH_FORWARDS_H_
#define UMINTL_LINE_SEARCH_FORWARDS_H_

#include <cmath>

#include "umintl/directions/forwards.h"
#include "umintl/directions/quasi_newton.hpp"

#include "umintl/optimization_context.hpp"


namespace umintl{

  inline double cubicmin(double a,double b, double fa, double fb, double dfa, double dfb, double xmin, double xmax){
    double d1 = dfa + dfb - 3*(fa - fb)/(a-b);
    double delta = pow(d1,2) - dfa*dfb;
    if(delta<0)
      return (xmin+xmax)/2;
    double d2 = std::sqrt(delta);
    double x = b - (b - a)*((dfb + d2 - d1)/(dfb - dfa + 2*d2));
    if(std::isnan(x))
      return (xmin+xmax)/2;
    return std::min(std::max(x,xmin),xmax);
  }

  inline double cubicmin(double a,double b, double fa, double fb, double dfa, double dfb){
    return cubicmin(a,b,fa,fb,dfa,dfb,std::min(a,b), std::max(a,b));
  }


  
  struct line_search_result{
    private:
      //NonCopyable, we do not want useless temporaries here
      line_search_result(line_search_result const &) : best_x(0, atidlas::FLOAT_TYPE), best_g(0, atidlas::FLOAT_TYPE){ }
      line_search_result & operator=(line_search_result const &){ }
    public:
      line_search_result(std::size_t dim) : has_failed(false), best_x(dim, atidlas::FLOAT_TYPE), best_g(dim, atidlas::FLOAT_TYPE){ }
      bool has_failed;
      double best_alpha;
      double best_phi;
      atidlas::array best_x;
      atidlas::array best_g;
  };

  
  struct line_search{
      
      line_search(unsigned int _max_evals) : max_evals(_max_evals){ }
      virtual ~line_search(){ }
      virtual void init(optimization_context &){ }
      virtual void clean(optimization_context &){ }
      virtual void operator()(line_search_result & res,umintl::direction * direction, optimization_context & context) = 0;
  protected:
      unsigned int max_evals;
  };


}

#endif
