/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_
#define UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "atidlas/array.h"
#include "umintl/tools/shared_ptr.hpp"
#include "umintl/optimization_context.hpp"

#include "forwards.h"



namespace umintl{


struct quasi_newton : public direction
{
    virtual std::string info() const
    { return "Quasi-Newton"; }

    virtual void clean(optimization_context &)
    {
      if(pH_.get())
        pH_.reset();
    }

    void operator()(optimization_context & c)
    {
      atidlas::array s(c.x() - c.xm1());
      atidlas::array y(c.g() - c.gm1());

      double ys = atidlas::value_scalar(atidlas::dot(s, y));

      if(pH_.get()==NULL)
      {
        std::size_t N = c.N();
        atidlas::numeric_type dtype = c.dtype();
        pH_.reset(new atidlas::array(atidlas::diag(N, dtype)));
      }
      atidlas::array& H = *pH_;

      atidlas::array Hy(c.N(), c.dtype());
      double gamma = 1;

      {
          Hy = atidlas::dot(H, y);
          double yHy = atidlas::value_scalar(atidlas::dot(y, Hy));
          double sg = atidlas::value_scalar(atidlas::dot(s, c.gm1()));
          double gHy = atidlas::value_scalar(atidlas::dot(c.gm1(), Hy));
          if(ys/yHy>1)
            gamma = ys/yHy;
          else if(sg/gHy<1)
            gamma = sg/gHy;
          else
            gamma = 1;
      }

      H*=gamma;
      Hy = atidlas::dot(H, y);
      double yHy = atidlas::value_scalar(atidlas::dot(y, Hy));

      //quasi_newton UPDATE
      double alpha = -1/ys;
      double beta = 1/ys + yHy/pow(ys,2);
      H += alpha*(atidlas::outer(s, Hy) + atidlas::outer(Hy, s)) + beta*atidlas::outer(s, s);
      c.p() = - atidlas::dot(H, c.g());
    }

private:
    tools::shared_ptr<atidlas::array> pH_;
};


}

#endif
