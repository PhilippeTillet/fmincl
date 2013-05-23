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

        struct state_ref{
            state_ref(unsigned int & _iter, viennacl::vector<double> & _xk, double & _valk, double & _valkm1
                        , viennacl::vector<double> & _gk, double & _dphi_0
                        , viennacl::vector<double> & _pk) : iter(_iter), x(_xk), val(_valk), valm1(_valkm1), g(_gk), dphi_0(_dphi_0), p(_pk){ }
            unsigned int & iter;
            viennacl::vector<double> & x;
            double & val;
            double & valm1;
            viennacl::vector<double> & g;
            double & dphi_0;
            viennacl::vector<double> & p;
        };

        class direction_base{
        public:
            virtual void operator()(detail::state_ref & state) = 0;
        };

        class line_search_base{
        public:
            virtual std::pair<double, bool> operator()(detail::state_ref & state) = 0;
        };



    }


}
#endif
