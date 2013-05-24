/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_LINE_SEARCH_INTERPOLATE_HPP_
#define FMINCL_LINE_SEARCH_INTERPOLATE_HPP_

#include <cmath>

namespace fmincl{

    namespace line_search{

        namespace interpolator{


            inline double cubicmin(double a,double b, double fa, double fb, double dfa, double dfb){
                double eps = 1e-3;
                double bma = b - a;
                double fab = (fb - fa)/bma;
                double d1 = dfa + dfb - 3*fab;
                double delta = pow(d1,2) - dfa*dfb;
                if(delta>=0){
                    double x;
                    double d2 = std::sqrt(delta);
                    double faab = (fab - dfa)/bma;
                    double faabb = (dfb - 2*fab + dfa)/pow(bma,2);
                    if(std::abs(faabb)<eps){
                        if(std::abs(faab)<eps)
                            x=a;
                        else
                            x= a - dfa/(2*faab);
                    }
                    else{
                        x = b - bma*(dfb + d2 - d1)/(dfb - dfa + 2*d2);
                    }
                    x = std::max(a,std::min(x,b));
                    double fx = (x-a)*(dfa + (x-a)*(faab + (x-b)*faabb));
                    if(fa <= fb && fa <= fx)
                        return a;
                    if(fb <= fx)
                        return b;
                    return x;
                }
                if(fa <= fb)
                    return a;
                return b;
            }


        }

    }

}

#endif
