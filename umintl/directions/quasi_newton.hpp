/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_
#define UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "isaac/array.h"
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
      isaac::array s(c.x() - c.xm1());
      isaac::array y(c.g() - c.gm1());

      double ys = isaac::value_scalar(dot(s, y));

      if(pH_.get()==NULL)
        pH_.reset(new isaac::array(isaac::eye(c.N(), c.N(), c.dtype())));
      isaac::array& H = *pH_;

      isaac::array Hy(c.N(), c.dtype());
      double gamma = 1;

      {
        Hy = isaac::dot(H, y);
        double yHy = isaac::value_scalar(dot(y, Hy));
        double sg = isaac::value_scalar(dot(s, c.gm1()));
        double gHy = isaac::value_scalar(dot(c.gm1(), Hy));
        if(ys/yHy>1)
          gamma = ys/yHy;
        else if(sg/gHy<1)
          gamma = sg/gHy;
        else
          gamma = 1;
      }

      H*=gamma;
      Hy = isaac::dot(H, y);
      double yHy = isaac::value_scalar(dot(y, Hy));

      //quasi_newton UPDATE
      double alpha = -1/ys;
      double beta = 1/ys + yHy/pow(ys,2);
      H += alpha*(isaac::outer(s, Hy) + isaac::outer(Hy, s)) + beta*isaac::outer(s, s);
      c.p() = - isaac::dot(H, c.g());
    }

  private:
    tools::shared_ptr<isaac::array> pH_;
  };


}

#endif
