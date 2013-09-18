/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_
#define FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_

#include "fmincl/utils.hpp"

#include "fmincl/mapping.hpp"

#include "fmincl/tools/shared_ptr.hpp"
#include "fmincl/directions/forwards.h"

#include "conjugate_gradient/restarts.hpp"
#include "conjugate_gradient/updates.hpp"

namespace fmincl{

struct conjugate_gradient : public direction{
    template<class BackendType>
    class implementation : public direction::implementation<BackendType>{
        typedef typename BackendType::VectorType VectorType;
        typedef implementation_of<BackendType,cg_update,polak_ribiere,fletcher_reeves,gilbert_nocedal> update_mapping;


    public:
        implementation(conjugate_gradient const & cg_params, detail::optimization_context<BackendType> & context) : context_(context), update_implementation_(update_mapping::create(*cg_params.update, context)){ }

        virtual bool restart(detail::optimization_context<BackendType> & c){
            double orthogonality_threshold = 0.1;
            return std::abs(BackendType::dot(c.dim(),c.g(),c.gm1()))/BackendType::dot(c.dim(),c.g(),c.g()) > orthogonality_threshold;
        }

        void operator()(detail::optimization_context<BackendType> & c){
          //p = -g + beta*p;
          double beta = (*update_implementation_)(c);
          BackendType::scale(c.dim(),beta,c.p());
          BackendType::axpy(c.dim(),-1,c.g(),c.p());
        }
    private:
        detail::optimization_context<BackendType> & context_;

        tools::shared_ptr<cg_update::implementation<BackendType> > update_implementation_;
    };


    conjugate_gradient(cg_update * _update = new polak_ribiere()) : update(_update){ }
    tools::shared_ptr<cg_update> update;
};

}

#endif
