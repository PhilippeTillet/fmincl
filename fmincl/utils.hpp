/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_UTILS_HPP
#define FMINCL_UTILS_HPP

#include <viennacl/forwards.h>

namespace fmincl{

    namespace detail{

        struct function_wrapper{
            virtual double operator()(viennacl::vector<double> const & x, viennacl::vector<double> * grad) const = 0;
        };

        template<class Fun>
        struct function_wrapper_impl : function_wrapper{
            function_wrapper_impl(Fun const & fun) : fun_(fun){ }
            double operator()(viennacl::vector<double> const & x, viennacl::vector<double> * grad) const { return fun_(x, grad); }
        private:
            Fun const & fun_;
        };

        class state{
        public:
            state(viennacl::vector<double> const & x0, detail::function_wrapper const & fun) : fun_(fun), iter_(0), dim_(x0.size()), x_(x0), g_(dim_), p_(dim_){

            }

            detail::function_wrapper const & fun() { return fun_; }
            unsigned int & iter() { return iter_; }
            unsigned int & dim() { return dim_; }
            viennacl::vector<double> & x() { return x_; }
            viennacl::vector<double> & g() { return g_; }
            viennacl::vector<double> & p() { return p_; }
            double & val() { return valk_; }
            double & valm1() { return valkm1_; }
            double & diff() { return diff_; }
            double & dphi_0() { return dphi_0_; }

        private:
            detail::function_wrapper const & fun_;
            unsigned int iter_;
            unsigned int dim_;
            viennacl::vector<double> x_;
            viennacl::vector<double> g_;
            viennacl::vector<double> p_;
            double valk_;
            double valkm1_;
            double diff_;
            double dphi_0_;
        };

        class direction_base{
        public:
            virtual void operator()(detail::state & state) = 0;
        };

        class line_search_base{
        public:
            virtual std::pair<double, bool> operator()(detail::state & state, double a_init) = 0;
        };

    }

    template<class FUN>
    void check_grad(FUN const & fun, viennacl::vector<double> const & x0){
        unsigned int dim = x0.size();
        viennacl::vector<double> x(x0);
        viennacl::vector<double> fgrad(dim);
        viennacl::vector<double> numgrad(dim);
        double eps = 1e-8;
        fun(x,&fgrad);
        for(unsigned int i=0 ; i < dim ; ++i){
            double old = x(i);
            x(i) = old-eps; double vleft = fun(x,NULL);
            x(i) = old+eps; double vright = fun(x,NULL);
            numgrad(i) = (vright-vleft)/(2*eps);
        }
        std::cout << numgrad - fgrad << std::endl;
    }


}
#endif
