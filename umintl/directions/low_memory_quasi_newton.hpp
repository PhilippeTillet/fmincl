/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_LOW_MEMORY_QUASI_NEWTON_HPP_
#define UMINTL_DIRECTIONS_LOW_MEMORY_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "atidlas/array.h"
#include "umintl/tools/shared_ptr.hpp"
#include "umintl/optimization_context.hpp"

#include "forwards.h"

namespace umintl{


  struct low_memory_quasi_newton : public direction{
    low_memory_quasi_newton(unsigned int _m = 4) : m(_m), n_valid_pairs_(0){ }
    unsigned int m;

  private:

    struct storage_pair{
      storage_pair(std::size_t N, atidlas::numeric_type dtype) : s(atidlas::zeros(N, 1, dtype)), y(atidlas::zeros(N, 1, dtype))
      { }
      atidlas::array s;
      atidlas::array y;
    };

    atidlas::array & s(std::size_t i) { return memory_[i].s; }
    atidlas::array & y(std::size_t i) { return memory_[i].y; }

  public:

    virtual std::string info() const
    { return "Low memory quasi-newton"; }

    void operator()(optimization_context & c)
    {
      if(n_valid_pairs_==0)
      {
        memory_.clear();
        memory_.reserve(m);
        for(unsigned int i = 0 ; i < m ; ++i)
          memory_.push_back(storage_pair(c.N(), c.dtype()));
      }

      std::vector<double> rhos(m);
      std::vector<double> alphas(m);

      //Algorithm
      n_valid_pairs_ = std::min(n_valid_pairs_+1,m);

      //Updates storage
      for(unsigned int i = n_valid_pairs_-1 ; i > 0  ; --i){
        s(i) = s(i-1);
        y(i) = y(i-1);
      }

      //s(0) = x - xm1;
      s(0) = c.x() - c.xm1();
      y(0) = c.g() - c.gm1();
      atidlas::array q = c.g();

      int i = 0;
      for(; i < (int)n_valid_pairs_ ; ++i){
        rhos[i] = atidlas::value_scalar(1/dot(y(i),s(i)));
        alphas[i] = atidlas::value_scalar(rhos[i]*dot(s(i), q));
        q = q - alphas[i]*y(i);
      }
      double scale = atidlas::value_scalar(dot(s(0), y(0))/dot(y(0),y(0)));

      atidlas::array r(scale*q);
      --i;
      for(; i >=0 ; --i){
        double beta = atidlas::value_scalar(rhos[i]*dot(y(i), r));
        r = r + (alphas[i] - beta)*s(i);
      }

      c.p() = -r;
    }

  private:
    std::vector<storage_pair> memory_;
    unsigned int n_valid_pairs_;
  };

}

#endif
