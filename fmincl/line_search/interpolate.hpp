#ifndef FMINCL_LINE_SEARCH_INTERPOLATE_HPP_
#define FMINCL_LINE_SEARCH_INTERPOLATE_HPP_


#include "utils.hpp"

namespace fmincl{

    namespace line_search{

        namespace interpolator{


            class cubic{
            public:
                cubic(double const & phi_0, double const & dphi_0) : phi_0_(phi_0), dphi_0_(dphi_0){ }
                template<class Fun>
                double operator()(double a0, double phi_a0, double a1, double phi_a1) const {
                    double x = phi_a1 - phi_0_ - dphi_0_*a1;
                    double y = phi_a0 - phi_0_ - dphi_0_*a0;
                    double norm = 1/(a0*a0*a1*a1*(a1-a0));
                    double a = 1/norm * (pow(a0,2)* x - pow(a1,2)*y);
                    double b = 1/norm * (-pow(a0,3)*x + pow(a1,3)*y);
                    double res = (-b + sqrt(pow(b,2) - 3*a*dphi_0_))/(3*a);
                    return res;
                }
            private:
                double const & phi_0_;
                double const & dphi_0_;
            };

            class quadratic{
            public:
                quadratic(double const & phi_0, double const & dphi_0) : phi_0_(phi_0), dphi_0_(dphi_0){ }
                template<class Fun>
                double operator()(double a0, double phi_a0){
                    double res = - dphi_0_*pow(a0,2)/(2*(phi_a0 - phi_0_ - dphi_0_*a0));
                }
            private:
                double const & phi_0_;
                double const & dphi_0_;
            };

            inline void safeguard(double const & aim1, double & ai, double eps=1e-4){
                if(abs(ai - aim1) < eps)
                    ai = 0.5*aim1;
            }

        }

    }

}

#endif
