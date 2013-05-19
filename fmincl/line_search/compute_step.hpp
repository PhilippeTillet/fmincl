#ifndef FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_
#define FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include "interpolate.hpp"
#include "termination.hpp"
#include "utils.hpp"

namespace fmincl{

    namespace line_search{

        template<class FUN>
        class step_computer{
        private:
            double zoom(double alo, double ahi, strong_wolf_powell const & termination, viennacl::vector<double> const & x, viennacl::vector<double> const & p) const{
                viennacl::vector<double> xi(x.size());
                viennacl::vector<double> grad(x.size());
                double phi_alo, phi_ahi, dphi_alo, dphi_ahi;
                double aj, phi_aj, dphi_aj;
                while(1){
                    xi = x + alo*p; viennacl::ocl::get_queue().finish(); phi_alo = fun_(xi, &grad); dphi_alo = viennacl::linalg::inner_prod(grad,p);
                    xi = x + ahi*p; viennacl::ocl::get_queue().finish(); phi_ahi = fun_(xi, &grad); dphi_ahi = viennacl::linalg::inner_prod(grad,p);
                    if(alo < ahi)
                        aj = interpolator::cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi);
                    else
                        aj = interpolator::cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo);
                    xi = x + aj*p; viennacl::ocl::get_queue().finish(); phi_aj = fun_(xi, NULL);
                    if(!termination.sufficient_decrease(aj,phi_aj) || phi_aj >= phi_alo){
                        ahi = aj;
                    }
                    else{
                        phi_aj = fun_(xi, &grad); dphi_aj = viennacl::linalg::inner_prod(grad,p);
                        if(termination.curvature(dphi_aj))
                            return aj;
                        if(dphi_aj*(ahi - alo) >= 0)
                            ahi = alo;
                        alo = aj;
                    }
                }
            }

        public:
            step_computer(FUN const & fun) : fun_(fun){ }
            double operator()(double phi_0
                              , double dphi_0
                              , double ai
                              , viennacl::vector<double> const & x
                              , viennacl::vector<double> const & p) const{
                size_t dim = x.size();
                double aim1 = 0;
                double phi_aim1 = phi_0;
                double dphi_aim1 = dphi_0;
                strong_wolf_powell termination(phi_0, dphi_0);
                double amax = 2;
                viennacl::vector<double> gi(dim);
                viennacl::vector<double> xi(dim);
                double phi_ai, dphi_ai;
                for(unsigned int i = 1 ; i<10 ; ++i){
                    xi = x + ai*p;
                    phi_ai = fun_(xi, NULL);

                    //Tests sufficient decrease
                    if(!termination.sufficient_decrease(ai, phi_ai) || (i>1 && phi_ai >= phi_aim1))
                        return zoom(aim1, ai, termination, x, p);
                    fun_(xi, &gi);
                    dphi_ai = viennacl::linalg::inner_prod(gi,p);

                    //Tests curvature
                    if(termination.curvature(dphi_ai))
                        return ai;
                    if(dphi_ai>=0)
                        return zoom(ai, aim1, termination, x, p);

                    //Updates states
                    aim1 = ai;
                    phi_aim1 = phi_ai;
                    dphi_aim1 = dphi_ai;
                    ai = 1.4*ai;
                    if(ai>amax || ai<1e-4)
                        return amax;
                }
            }
        private:
            FUN const & fun_;
        };

    }

}

#endif
