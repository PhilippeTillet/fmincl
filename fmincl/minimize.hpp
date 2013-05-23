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

    template<class Fun>
    class line_search_proxy{
    public:
        line_search_proxy(Fun const & fun) : fun_(fun){ }

        template<class U>
        line_search_proxy<Fun> & operator=(U const & u){ ptr = new typename result_of::tag_to_line_search<U, Fun>::type(fun_, u); }

        detail::line_search_base & get(){
            return *ptr;
        }
    private:
        Fun const & fun_;
        detail::line_search_base * ptr;
    };

    class direction_proxy{
    public:
        template<class U>
        direction_proxy & operator=(U const & u){ ptr = new typename result_of::tag_to_direction<U>::type(u); }

        detail::direction_base & get(){
            return *ptr;
        }
    private:
        detail::direction_base * ptr;
    };


    template<class FUN>
    class minimizer{
    public:
        minimizer(FUN const & fun) : fun_(fun), line_search(fun){ }

        viennacl::vector<double> operator()(viennacl::vector<double> const & x0){
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
    //        line_search::strong_wolfe_powell step(fun, line_search::strong_wolfe_powell_tag(1e-4, 0.1,1.4));

//            direction::quasi_newton update_dir;
//            line_search::strong_wolfe_powell_tag tag(1e-4, 0.9,1.4);

//            typename result_of::type_of_tag<line_search::strong_wolfe_powell_tag, FUN>::type get_step_size(fun_,tag);

            for( ; iter < max_iter ; ++iter){
                valk = fun_(x, &gk);
                if(iter>0) std::cout << "iter " << iter << " | cost : " << valk << std::endl;
                diff = (valk-valkm1);
                viennacl::backend::finish();
                direction.get()(state);
                dphi_0 = viennacl::linalg::inner_prod(pk,gk);
                if(dphi_0>0){
                    pk = -gk;
                    dphi_0 = - viennacl::linalg::inner_prod(gk,gk);
                }
                std::pair<double, bool> search_res = line_search.get()(state);
                if(search_res.second) break;
                x = x + search_res.first*pk;
                valkm1 = valk;
            }
            return x;
        }

        direction_proxy direction;
        line_search_proxy<FUN> line_search;

    private:
        FUN const & fun_;
    };

}

#endif
