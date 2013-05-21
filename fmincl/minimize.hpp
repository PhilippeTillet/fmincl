#ifndef FMINCL_MINIMIZE_HPP_
#define FMINCL_MINIMIZE_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include "fmincl/directions/cg.hpp"
#include "fmincl/directions/quasi-newton.hpp"
#include "fmincl/line_search/compute_step.hpp"

namespace fmincl{

    template<class FUN>
    viennacl::vector<double> minimize(FUN const & fun, viennacl::vector<double> const & x0){
        std::cout << "Start at : " << x0 << std::endl;
        viennacl::vector<double> x = x0;
        unsigned int max_iter = 2000;
        unsigned int dim = x.size();
        viennacl::vector<double> gk(dim);
        viennacl::vector<double> pk(dim);
        double valk, valkm1, diff, dphi_0, initial_alpha;
        //direction::cg<direction::tags::polak_ribiere,direction::tags::no_restart> update_dir(pk,gk);
        direction::quasi_newton update_dir(pk, x, gk);
        line_search::strong_wolfe_powell<FUN> strong_wolfe_powell_step(fun, valk, dphi_0, 1e-4, 0.9);
        for(unsigned int i=0 ; i < max_iter ; ++i){
            valk = fun(x, &gk);
            if(i>0) std::cout << "iter " << i << " | cost : " << valk << std::endl;
            diff = (valk-valkm1);
            viennacl::backend::finish();
            update_dir();
            dphi_0 = viennacl::linalg::inner_prod(pk,gk);
            if(dphi_0>0){
                pk = -gk;
                dphi_0 = - viennacl::linalg::norm_2(gk);
            }
            initial_alpha = (i==0)?1:std::min(1.0d,1.01*2*diff/dphi_0);
            std::pair<double, bool> search_res = strong_wolfe_powell_step(initial_alpha,x,pk);
            if(search_res.second) break;
            x = x + search_res.first*pk;
            valkm1 = valk;
        }
        return x;
    }
}

#endif
