#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

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
#include "fmincl/utils.hpp"

#include "mghfuns/beale.hpp"
#include "mghfuns/powell_badly_scaled.hpp"
#include "mghfuns/brown_badly_scaled.hpp"
#include "mghfuns/rosenbrock.hpp"
#include "mghfuns/helical_valley.hpp"
#include "mghfuns/biggs_exp6.hpp"
#include "mghfuns/watson.hpp"
#include "mghfuns/variably_dimensioned.hpp"
#include "mghfuns/box_3d.hpp"
#include "mghfuns/gaussian.hpp"
#include "mghfuns/penalty1.hpp"
#include "mghfuns/penalty2.hpp"
#include "mghfuns/gulf.hpp"
#include "mghfuns/powell_singular.hpp"
#include "mghfuns/wood.hpp"
#include "mghfuns/trigonometric.hpp"
#include "mghfuns/brown_dennis.hpp"
#include "mghfuns/gaussian.hpp"

using namespace fmincl;

template<class ScalarType>
struct get_backend{
    typedef fmincl::backend::cblas_types<ScalarType> type;
};


template<class FunctionType>
int test_function(FunctionType const & fun, fmincl::optimization_options const & options)
{
    typedef typename FunctionType::BackendType BackendType;
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;

    int res = EXIT_SUCCESS;
    ScalarType epsilon = 1e-4;
    ScalarType diff = 0;

    std::cout << "- Testing " << fun.name() << "..." << std::flush;

    std::size_t dimension = fun.N();
    VectorType X0 = BackendType::create_vector(dimension);
    fun.init(X0);

    ScalarType true_minimum = fun.global_minimum();

    //fmincl::utils::check_grad<BackendType>(FunctionType(),X0,dimension);
    VectorType S = BackendType::create_vector(dimension);
    fmincl::optimization_result result = fmincl::minimize<BackendType>(S,fun,X0,dimension,options);

    ScalarType numerical_minimum = (ScalarType)result.f;

    diff = std::fabs(fun.global_minimum() - numerical_minimum);

    if(true_minimum>1)
        diff/=std::max(true_minimum,numerical_minimum);

    if(diff>epsilon){ \
        std::vector<ScalarType> local_minima = fun.local_minima();
        ScalarType min_local_minima_diff = INFINITY;
        for(typename std::vector<ScalarType>::iterator it = local_minima.begin() ; it != local_minima.end() ; ++it){
            ScalarType new_diff = std::fabs(numerical_minimum - *it);
            if(*it>1)
                new_diff/=std::max(numerical_minimum,*it);
            min_local_minima_diff = std::min(min_local_minima_diff,new_diff);
        }

        if(min_local_minima_diff<=epsilon){
#ifndef DISABLE_WARNING
            std::cout << "#Warning for " << function_name << " : Converge to local minimum!" << std::flush ;
#endif
            std::cout << std::endl;
        }
        else{
            if(min_local_minima_diff<diff)
                std::cout << " Fail! /* Diff = " << min_local_minima_diff << "*/" << std::endl; \
            else
                std::cout << " Fail! /* Diff = " << diff << "*/" << std::endl; \
            res = EXIT_FAILURE;
        }\

    }
    else
        std::cout << std::endl;

    BackendType::delete_if_dynamically_allocated(X0);
    BackendType::delete_if_dynamically_allocated(S);

    return res;
}
template<class ScalarType>
int test_option(std::string const & options_name, fmincl::direction * direction){
    typedef fmincl::backend::cblas_types<ScalarType> BackendType;
    const std::size_t max_iter = 4096;
    const unsigned int verbosity = 0;
    optimization_options options(direction, new gradient_treshold(), max_iter, verbosity);
    std::cout << "Testing " << options_name << "..." << std::endl;
    int res = EXIT_SUCCESS;
    res |= test_function(helical_valley<BackendType>(),options);
    res |= test_function(biggs_exp6<BackendType>(),options);
    res |= test_function(gaussian<BackendType>(),options);
    res |= test_function(powell_badly_scaled<BackendType>(),options);
    res |= test_function(box_3d<BackendType>(),options);
    res |= test_function(variably_dimensioned<BackendType>(20),options);
    res |= test_function(watson<BackendType>(6),options);
    res |= test_function(penalty1<BackendType>(10),options);
    res |= test_function(penalty2<BackendType>(10),options);
    res |= test_function(brown_badly_scaled<BackendType>(),options);
    res |= test_function(brown_dennis<BackendType>(),options);
    res |= test_function(gulf<BackendType>(20),options);
    res |= test_function(trigonometric<BackendType>(10),options);
    res |= test_function(rosenbrock<BackendType>(2),options);
    res |= test_function(powell_singular<BackendType>(4),options);
    res |= test_function(rosenbrock<BackendType>(20),options);
    res |= test_function(powell_singular<BackendType>(40),options);

//    res |= test_function(beale<BackendType>(),options);
//    res |= test_function(wood<BackendType>(),options);
    //res |= test_function(chebyquad<BackendType>(),options);

    return res;
}

#endif
