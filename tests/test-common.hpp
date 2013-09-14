#ifndef FMINCL_TEST_COMMON_HPP_
#define FMINCL_TEST_COMMON_HPP_

/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include <iostream>

#include "fmincl/backends/cblas.hpp"
#include "fmincl/minimize.hpp"

#include "common/rosenbrock.hpp"
#include "common/beale.hpp"

static const float epsilon = 1e-6;

template<class ScalarType>
struct get_backend{
    typedef fmincl::backend::cblas_types<ScalarType> type;
};

template<class BackendType>
double test_result(std::size_t N, typename BackendType::VectorType const & S, typename BackendType::VectorType const & RealS){
    typedef typename BackendType::ScalarType ScalarType;
    ScalarType diff = 0;
    for(std::size_t i = 0 ; i < N ; ++i){
        if(std::isnan(S[i])) return INFINITY;
        diff = std::max(diff,std::fabs(S[i]-RealS[i]));
    }
    return diff;
}

template<class BackendType, class FunctionType>
int test_function(std::string const & function_name, fmincl::optimization_options const & options)
{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    ScalarType diff = 0;
    static const std::size_t dimension = FunctionType::N;
    VectorType X0 = BackendType::create_vector(dimension);
    VectorType S = BackendType::create_vector(dimension);
    std::cout << "    * Testing " << function_name << "..." << std::endl;
    if((diff = std::fabs(FunctionType::true_minimum_value() - fmincl::minimize<BackendType>(S,FunctionType(),X0,dimension,options)))>epsilon){ \
        std::cout << "## Failure! Diff = " << diff << std::endl; \
        return EXIT_FAILURE; \
    }
    BackendType::delete_if_dynamically_allocated(X0);
    BackendType::delete_if_dynamically_allocated(S);
    return EXIT_SUCCESS;
}
template<class BackendType>
int test_option(std::string const & options_name, fmincl::optimization_options const & options){
    std::cout << "  Testing " << options_name << "..." << std::endl;
    int res = EXIT_SUCCESS;
    res |= test_function<BackendType,rosenbrock<BackendType> >("Rosenbrock",options);
    res |= test_function<BackendType,beale<BackendType> >("Beale",options);
    return res;
}

#endif
