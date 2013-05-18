#ifndef FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_
#define FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include "interpolate.hpp"
#include "termination.hpp"
#include "utils.hpp"

namespace fmincl{

    namespace line_search{

        template<class FUN, class INTERPOLATOR, class TERMINATION>
        class step_computer{
        private:
            double zoom(double alo, double phi_alo, double ahi, double phi_ahi, INTERPOLATOR const & interpolator, TERMINATION const & termination
                        ,viennacl::vector<double> & xi, viennacl::vector<double> const & x, viennacl::vector<double> const & p, viennacl::vector<double> * g){
                double aj, phi_aj, dphi_aj;
                aj = interpolator(alo,phi_alo,ahi,phi_ahi);
                xi = x + aj*p;
                phi_aj = fun_(xi);
                if(!termination.sufficient_decrease(aj,phi_aj) || phi_aj >= phi_alo){
                    ahi = aj;
                }
                else{
                    fun_(xi, g);
                    dphi_aj = viennacl::linalg::inner_prod(g,p);
                    if(termination.curvature(dphi_aj))
                        return aj;
                    if(dphi_aj*(ahi - alo) >= 0)
                        ahi = alo;
                    alo = aj;
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
                INTERPOLATOR interpolator(phi_0, dphi_0);
                TERMINATION termination(phi_0, dphi_0);
                double amax = 2;
                unsigned int max_i = 10;
                viennacl::vector<double> gi(dim);
                viennacl::vector<double> xi(dim);
                double phi_ai, dphi_ai;
                while(1){
                    xi = x + ai*p;
                    phi_ai = fun_(xi);

                    //Tests sufficient decrease
                    if(!termination.sufficient_decrease(ai, phi_ai) || (i>1 && phi_ai >= phi_aim1))
                        return zoom(aim1,phi_aim1, ai, phi_ai, x, p, &gi);
                    fun_(xi, &gi);
                    dphi_ai = viennacl::linalg::inner_prod(gi,p);

                    //Tests curvature
                    if(termination.curvature(dphi_ai))
                        return ai;
                    if(dphi_ai>=0)
                        return zoom(ai, phi_ai, aim1, phi_aim1, interpolator, x, p &gi);

                    //Updates states
                    aim1 = ai;
                    phi_aim1 = phi_ai;
                    dphi_aim1 = dphi_ai;

                    ai = 0.5*(aim1+amax);
                }
            }
        private:
            FUN const & fun_;
        };

    }

}
