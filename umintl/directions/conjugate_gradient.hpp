/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_
#define UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_

#include "umintl/optimization_context.hpp"
#include "umintl/tools/shared_ptr.hpp"
#include "umintl/directions/forwards.h"
#include "atidlas/array.h"

namespace umintl{

namespace tag{

namespace conjugate_gradient{

enum restart{
    NO_RESTART,
    RESTART_ON_DIM,
    RESTART_NOT_ORTHOGONAL
};

enum update{
    UPDATE_POLAK_RIBIERE,
    UPDATE_GILBERT_NOCEDAL,
    UPDATE_FLETCHER_REEVES
};

}

}


struct conjugate_gradient : public direction{
public:


private:
    atidlas::array_expression update_polak_ribiere(optimization_context & c)
    { return atidlas::max(dot(c.g(), c.g() - c.gm1())/dot(c.gm1(), c.gm1()), 0); }

    atidlas::array_expression update_fletcher_reeves(optimization_context & c)
    { return atidlas::dot(c.g(), c.g())/atidlas::dot(c.gm1(), c.gm1()); }

    atidlas::array_expression update_impl(optimization_context & c){
        switch (update) {
            case tag::conjugate_gradient::UPDATE_POLAK_RIBIERE: return update_polak_ribiere(c);
        case tag::conjugate_gradient::UPDATE_GILBERT_NOCEDAL: return atidlas::min(update_polak_ribiere(c), update_fletcher_reeves(c));
            case tag::conjugate_gradient::UPDATE_FLETCHER_REEVES: return update_fletcher_reeves(c);
            default: throw exceptions::incompatible_parameters("Unsupported conjugate gradient update");
        }
    }

    bool restart_on_dim(optimization_context & c){
        return c.iter()==c.N();
    }

    bool restart_not_orthogonal(optimization_context & c){
        double threshold = 0.1;
        double ratio = atidlas::value_scalar(atidlas::abs(dot(c.g(), c.gm1()))/atidlas::dot(c.g(), c.g()));
        return ratio > threshold;
    }

    bool restart_impl(optimization_context & c){
        switch (restart) {
            case tag::conjugate_gradient::NO_RESTART: return false;
            case tag::conjugate_gradient::RESTART_ON_DIM: return restart_on_dim(c);
            case tag::conjugate_gradient::RESTART_NOT_ORTHOGONAL: return restart_not_orthogonal(c);
            default: throw exceptions::incompatible_parameters("Unsupported conjugate gradient restart");
        }
    }



public:
    conjugate_gradient(tag::conjugate_gradient::update _update = tag::conjugate_gradient::UPDATE_POLAK_RIBIERE
            , tag::conjugate_gradient::restart _restart = tag::conjugate_gradient::RESTART_NOT_ORTHOGONAL) : update(_update), restart(_restart){ }

    virtual std::string info() const{
        return "Nonlinear Conjugate Gradient";
    }

    void operator()(optimization_context & c){
        double beta;
        if(restart_impl(c))
            beta = 0;
        else
            beta = atidlas::value_scalar(update_impl(c));
        c.p() = beta*c.p() - c.g();
    }

    tag::conjugate_gradient::update update;
    tag::conjugate_gradient::restart restart;
};

}

#endif
