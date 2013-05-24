/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_MINIMIZE_HPP_
#define FMINCL_MINIMIZE_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include "fmincl/directions/cg.hpp"
#include "fmincl/directions/quasi-newton.hpp"
#include "fmincl/line_search/strong_wolfe_powell.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

    template<class RETURN_TYPE, class Type>
    class proxy{
    public:
        template<class U>
        proxy & operator=(U const & u){ ptr = new U(u); }
        RETURN_TYPE operator()(detail::state_ref & state) const { return (*ptr)(state); }
    private:
        Type * ptr;
    };

    struct optimization_options{
        proxy<void, detail::direction_base> direction;
        proxy<std::pair<double, bool>, detail::line_search_base> line_search;
    };

    template<class Fun>
    viennacl::vector<double> minimize(Fun const & fun, viennacl::vector<double> const & x0, optimization_options const & options){
        detail::function_wrapper_impl<Fun> wrapper(fun);
        std::cout << "Start at : " << x0 << std::endl;
        viennacl::vector<double> x = x0;
        unsigned int max_iter = 2000;
        unsigned int dim = x.size();
        viennacl::vector<double> gk(dim);
        viennacl::vector<double> pk(dim);
        double valk, valkm1, diff, dphi_0;
        unsigned int iter=0;
        detail::state_ref state(iter, x, valk, valkm1, gk, dphi_0, pk, wrapper);
        for( ; iter < max_iter ; ++iter){
            valk = wrapper(x, &gk);
            if(iter>0) std::cout << "iter " << iter << " | cost : " << valk << std::endl;
            diff = (valk-valkm1);
            viennacl::backend::finish();
            options.direction(state);
            dphi_0 = viennacl::linalg::inner_prod(pk,gk);
            if(dphi_0>0){
                pk = -gk;
                dphi_0 = - viennacl::linalg::inner_prod(gk,gk);
            }
            std::pair<double, bool> search_res = options.line_search(state);
            if(search_res.second) break;
            x = x + search_res.first*pk;
            valkm1 = valk;
        }
        return x;
    }

}

#endif
