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
        double tol = 1e-4;
        viennacl::vector<double> x = x0;
        unsigned int max_iter = 100;
        viennacl::vector<double> gk(x.size());
        viennacl::vector<double> gkm1(x.size());
        viennacl::vector<double> pk(x.size());
        direction::cg<direction::tags::polak_ribiere,direction::tags::no_restart> compute_direction;
        line_search::step_computer<FUN> compute_step(fun);
        double valk, valkm1, diff, dphi_0, initial_alpha, ai;
        for(unsigned int i=0 ; i < max_iter ; ++i){
            valk = fun(x, &gk);
            diff = (valk-valkm1);
            compute_direction(pk, gk, (i==0)?NULL:&gkm1);
            dphi_0 = viennacl::linalg::inner_prod(pk,gk);
            initial_alpha = (i==0)?1:2*diff/dphi_0;
            std::pair<double, bool> search_res = compute_step(valk,dphi_0,initial_alpha,x,pk);
            if(search_res.second) break;
            x = x + search_res.first*pk;
            if(i>0) std::cout << "iter " << i << " | cost : " << valk << std::endl;
            valkm1 = valk;
            gkm1 = gk;
        }
        return x;
    }
}

#endif
