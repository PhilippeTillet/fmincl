#ifndef FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_
#define FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include "interpolate.hpp"

namespace fmincl{

    namespace line_search{

        template<class FUN>
        class strong_wolfe_powell{
        private:
            std::pair<double, bool> zoom(double alo, double ahi, viennacl::vector<double> const & x, viennacl::vector<double> const & p) const{
                viennacl::vector<double> xi(x.size());
                viennacl::vector<double> grad(x.size());
                double phi_alo, phi_ahi, dphi_alo, dphi_ahi;
                double aj, phi_aj, dphi_aj;
                while(1){
                    xi = x + alo*p; viennacl::backend::finish(); phi_alo = fun_(xi, &grad); dphi_alo = viennacl::linalg::inner_prod(grad,p);
                    xi = x + ahi*p; viennacl::backend::finish(); phi_ahi = fun_(xi, &grad); dphi_ahi = viennacl::linalg::inner_prod(grad,p);
                    if(alo < ahi)
                        aj = interpolator::cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi);
                    else
                        aj = interpolator::cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo);
                    if(aj==alo || aj==ahi){
                        return std::make_pair(ahi,true);
                    }
                    xi = x + aj*p; viennacl::backend::finish(); phi_aj = fun_(xi, NULL);
                    if(!sufficient_decrease(aj,phi_aj) || phi_aj >= phi_alo){
                        ahi = aj;
                    }
                    else{
                        phi_aj = fun_(xi, &grad); dphi_aj = viennacl::linalg::inner_prod(grad,p);
                        if(curvature(dphi_aj))
                            return std::make_pair(aj, false);
                        if(dphi_aj*(ahi - alo) >= 0)
                            ahi = alo;
                        alo = aj;
                    }
                }
            }

            bool sufficient_decrease(double ai, double phi_ai) const {
                return phi_ai <= (phi_0_ + c1_*ai*dphi_0_);
            }
            bool curvature(double dphi_ai) const{
                return std::abs(dphi_ai) <= c2_*std::abs(dphi_0_);
            }

        public:            
            strong_wolfe_powell(FUN const & fun, double const & phi_0, double const & dphi_0, double c1, double c2) : fun_(fun), phi_0_(phi_0), dphi_0_(dphi_0), c1_(c1), c2_(c2){ }

            std::pair<double, bool> operator()(double ai
                              , viennacl::vector<double> const & x
                              , viennacl::vector<double> const & p) const{
                size_t dim = x.size();
                double rho = 1.4;
                double aim1 = 0;
                double phi_aim1 = phi_0_;
                double dphi_aim1 = dphi_0_;
                double amax = 5;
                viennacl::vector<double> gi(dim);
                viennacl::vector<double> xi(dim);
                double phi_ai, dphi_ai;
                for(unsigned int i = 1 ; i<200; ++i){
                    xi = x + ai*p;
                    phi_ai = fun_(xi, NULL);

                    //Tests sufficient decrease
                    if(!sufficient_decrease(ai, phi_ai) || (i>1 && phi_ai >= phi_aim1))
                        return zoom(aim1, ai,  x, p);
                    fun_(xi, &gi);
                    dphi_ai = viennacl::linalg::inner_prod(gi,p);
                    //Tests curvature
                    if(curvature(dphi_ai))
                        return std::make_pair(ai, false);
                    if(dphi_ai>=0)
                        return zoom(ai, aim1, x, p);

                    //Updates states
                    aim1 = ai;
                    phi_aim1 = phi_ai;
                    dphi_aim1 = dphi_ai;
                    ai = rho*ai;
                    if(ai>amax)
                        return std::make_pair(amax,true);
                }
                return std::make_pair(amax,false);
            }
        private:
            FUN const & fun_;
            double const & phi_0_;
            double const & dphi_0_;
            double c1_;
            double c2_;
        };

    }

}

#endif
