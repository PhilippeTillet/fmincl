/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_OPTIMIZATION_CONTEXT_HPP
#define UMINTL_OPTIMIZATION_CONTEXT_HPP

#include "umintl/tools/shared_ptr.hpp"
#include "umintl/function_wrapper.hpp"
#include <iostream>

namespace umintl{

    /** @brief The optimization context class
     *
     *  Holds the state of the optimization procedure. Typically passed as function argument, to allow easy
     * access to the usual quantities of interest
     */
    
    class optimization_context{
    private:
        optimization_context(optimization_context const & other);
        optimization_context& operator=(optimization_context const & other);
    public:
        
        
        

        optimization_context(atidlas::array const & x0, std::size_t dim, model_base & model, detail::function_wrapper * fun) : fun_(fun), model_(model), iter_(0), dim_(dim),
                                                                                                                    dtype_(x_.dtype()), x_(x0), g_(dim_, dtype_),
                                                                                                                    p_(dim_, dtype_), xm1_(dim_, dtype_), gm1_(dim_, dtype_),
                                                                                                                    valk_(dtype_), valkm1_(dtype_), dphi_0_(dtype_), alpha_(dtype_)
        { }

        model_base & model(){ return model_; }
        atidlas::numeric_type dtype() const { return dtype_; }
        detail::function_wrapper & fun() { return *fun_; }
        unsigned int & iter() { return iter_; }
        unsigned int & N() { return dim_; }
        atidlas::array & x() { return x_; }
        atidlas::array & g() { return g_; }
        atidlas::array & xm1() { return xm1_; }
        atidlas::array & gm1() { return gm1_; }
        atidlas::array & p() { return p_; }
        double & val() { return valk_; }
        double & valm1() { return valkm1_; }
        double & dphi_0() { return dphi_0_; }
        double & alpha() { return alpha_; }

    private:
        tools::shared_ptr< detail::function_wrapper > fun_;
        model_base & model_;

        unsigned int iter_;
        unsigned int dim_;
        atidlas::numeric_type dtype_;

        atidlas::array x_;
        atidlas::array g_;
        atidlas::array p_;
        atidlas::array xm1_;
        atidlas::array gm1_;

        double valk_;
        double valkm1_;
        double dphi_0_;

        double alpha_;
    };
}
#endif
