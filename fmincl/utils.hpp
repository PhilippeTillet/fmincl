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

    }


}
#endif
