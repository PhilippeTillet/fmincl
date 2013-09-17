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
        typedef implementation_of<BackendType,cg_update,polak_ribiere,fletcher_reeves> update_mapping;


    public:
        implementation(conjugate_gradient const & cg_params, detail::optimization_context<BackendType> & context) : context_(context), update_implementation_(update_mapping::create(*cg_params.update, context)){ }
        void operator()(){
          double eps = 1e-9;
          VectorType const & g = context_.g();
          VectorType & p = context_.p();
          std::size_t N = context_.dim();
          bool restart = std::abs(BackendType::dot(N,g,context_.gm1()))/BackendType::dot(N,g,g) > 0.2;
          double beta = restart?0:(*update_implementation_)();
          //p = -g + beta*p;
          BackendType::scale(N,beta,p);
          BackendType::axpy(N,-1,g,p);
          //Restart


        }
    private:
        detail::optimization_context<BackendType> & context_;

        tools::shared_ptr<cg_update::implementation<BackendType> > update_implementation_;
    };


    conjugate_gradient(cg_update * _update = new fletcher_reeves()) : update(_update){ }
    tools::shared_ptr<cg_update> update;
};

}

#endif
