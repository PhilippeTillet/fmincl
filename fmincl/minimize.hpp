#ifndef FMINCL_MINIMIZE_HPP_
#define FMINCL_MINIMIZE_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include "fmincl/directions/cg.hpp"
#include "fmincl/line_search/compute_step.hpp"

namespace fmincl{

    template<class FUN>
    viennacl::vector<double> minimize(FUN const & fun, viennacl::vector<double> const & x0){
        std::cout << "Start at : " << x0 << std::endl;
        viennacl::vector<double> x = x0;
        unsigned int max_iter = 2000;
        unsigned int dim = x.size();
        viennacl::vector<double> gk(dim);
        viennacl::vector<double> gkm1(dim);
        viennacl::vector<double> pk(dim);
        direction::cg<direction::tags::polak_ribiere,direction::tags::no_restart> compute_direction;
        line_search::step_computer<FUN> compute_step(fun);
        double valk, valkm1, diff, dphi_0, initial_alpha, ai;
        for(unsigned int i=0 ; i < max_iter ; ++i){
            valk = fun(x, &gk);
            if(i>0) std::cout << "iter " << i << " | cost : " << valk << std::endl;
            diff = (valk-valkm1);
            viennacl::backend::finish();
            compute_direction(pk, gk, (i==0)?NULL:&gkm1);
            dphi_0 = viennacl::linalg::inner_prod(pk,gk);
            if(dphi_0>0){
                pk = -gk;
                dphi_0 = - viennacl::linalg::norm_2(gk);
            }
            initial_alpha = 1;
            std::pair<double, bool> search_res = compute_step(valk,dphi_0,initial_alpha,x,pk);
            if(search_res.second) break;
            x = x + search_res.first*pk;
            valkm1 = valk;
            gkm1 = gk;
        }
        return x;
    }
}

#endif
