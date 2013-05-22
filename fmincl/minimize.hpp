#ifndef FMINCL_MINIMIZE_HPP_
#define FMINCL_MINIMIZE_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include "fmincl/directions/cg.hpp"
#include "fmincl/directions/quasi-newton.hpp"
#include "fmincl/line_search/compute_step.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{


    template<class FUN>
    viennacl::vector<double> minimize(FUN const & fun, viennacl::vector<double> const & x0){
        std::cout << "Start at : " << x0 << std::endl;
        viennacl::vector<double> x = x0;
        unsigned int max_iter = 2000;
        unsigned int dim = x.size();
        viennacl::vector<double> gk(dim);
        viennacl::vector<double> pk(dim);
        double valk, valkm1, diff, dphi_0;
        unsigned int iter=0;
        detail::state_ref state(iter, x, valk, valkm1, gk, dphi_0, pk);
//        direction::cg<direction::tags::polak_ribiere,direction::tags::no_restart> update_dir;
//        line_search::strong_wolfe_powell<FUN> step(fun, 1e-4, 0.1);

        direction::quasi_newton update_dir;
        line_search::strong_wolfe_powell<FUN> step(fun, 1e-4, 0.9);

        for( ; iter < max_iter ; ++iter){
            valk = fun(x, &gk);
            if(iter>0) std::cout << "iter " << iter << " | cost : " << valk << std::endl;
            diff = (valk-valkm1);
            viennacl::backend::finish();
            update_dir(state);
            dphi_0 = viennacl::linalg::inner_prod(pk,gk);
            if(dphi_0>0){
                pk = -gk;
                dphi_0 = - viennacl::linalg::norm_2(gk);
            }
            std::pair<double, bool> search_res = step(state);
            if(search_res.second) break;
            x = x + search_res.first*pk;
            valkm1 = valk;
        }
        return x;
    }
}

#endif
