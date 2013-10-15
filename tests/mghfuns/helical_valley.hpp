/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_HELICAL_VALLEY_HPP_
#define UMINTL_HELICAL_VALLEY_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class helical_valley : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;

    ScalarType theta(ScalarType x, ScalarType y) const {
        if(x>0)
            return 1/(ScalarType)M_2_PI*atan(y/x);
        else
            return 1/(ScalarType)M_2_PI*atan(y/x) + 0.5;
    }

    ScalarType dtheta_dx(ScalarType x, ScalarType y) const {
       return -1/(ScalarType)M_2_PI*y/(y*y+x*x);
    }

    ScalarType dtheta_dy(ScalarType x, ScalarType y) const {
       return 1/(ScalarType)M_2_PI*x/(x*x+y*y);
    }

public:
    helical_valley() : base_type("Helical Valley",3,3,0){ }
    void init(VectorType & X) const
    {
        X[0] = -1;
        X[1] = 0;
        X[2] = 0;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        RES(0,0)= -100*dtheta_dx(V[0],V[1]);                  RES(0,1)= -100*dtheta_dy(V[0],V[1]);               RES(0,2)= 10;
        RES(1,0)= 10*V[0]/sqrt(V[0]*V[0] + V[1]*V[1]);        RES(1,1)= 10*V[1]/sqrt(V[0]*V[0] + V[1]*V[1]);     RES(1,2)= 0;
        RES(2,0)= 0;                                          RES(2,1)= 0;                                       RES(2,2)= 1;

    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        res[0] = 10*(V[2] - 10*theta(V[0],V[1]));
        res[1] = 10*(sqrt(V[0]*V[0] + V[1]*V[1]) - 1);
        res[2] = V[2];
    }
};

#endif
