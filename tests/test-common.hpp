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
#include "fmincl/utils.hpp"
#include "common/beale.hpp"
#include "common/freudenstein-roth.hpp"
#include "common/powell-badly-scaled.hpp"
#include "common/rosenbrock.hpp"

template<class ScalarType>
struct tolerance{
    static const ScalarType value;
};

template<> const float tolerance<float>::value = 5e-3;
template<> const double tolerance<double>::value = 1e-5;

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
    int res = EXIT_SUCCESS;
    ScalarType epsilon = tolerance<ScalarType>::value;
    ScalarType diff = 0;
    static const std::size_t dimension = FunctionType::N;
    VectorType X0 = BackendType::create_vector(dimension);
    FunctionType::init(X0);
    //fmincl::utils::check_grad<BackendType>(FunctionType(),X0,dimension);
    VectorType S = BackendType::create_vector(dimension);
    std::cout << "    * Testing " << function_name << "..." << std::flush;
    ScalarType found_minimum = fmincl::minimize<BackendType>(S,FunctionType(),X0,dimension,options);
    if((diff = std::fabs(FunctionType::true_minimum_value() - found_minimum))>epsilon){ \
        std::vector<ScalarType> local_minima;
        FunctionType::local_minima_value(local_minima);
        ScalarType min_local_minima_diff = INFINITY;
        for(typename std::vector<ScalarType>::iterator it = local_minima.begin() ; it != local_minima.end() ; ++it)
            min_local_minima_diff = std::min(min_local_minima_diff,std::fabs(found_minimum - *it));

        if(min_local_minima_diff<=epsilon)
            std::cout << "#Warning : Converge to local minimum!" << std::endl;
        else{
            if(min_local_minima_diff<diff)
                std::cout << "## Failure ! Closer to local minima ! Diff = " << min_local_minima_diff << std::endl; \
            else
                std::cout << "## Failure! Diff = " << diff << std::endl; \
            res = EXIT_FAILURE;
        }\
    }
    else{
        std::cout << "Passed !" << std::endl;
    }
    BackendType::delete_if_dynamically_allocated(X0);
    BackendType::delete_if_dynamically_allocated(S);
    return res;
}
template<class BackendType>
int test_option(std::string const & options_name, fmincl::optimization_options const & options){
    std::cout << "  Testing " << options_name << "..." << std::endl;
    int res = EXIT_SUCCESS;
    //res |= test_function<BackendType,beale<BackendType> >("Beale",options);
    res |= test_function<BackendType,rosenbrock<10,BackendType> >("Extended Rosenbrock",options);
    //res |= test_function<BackendType,freudenstein_roth<BackendType> >("Freudenstein-Roth",options);
    //res |= test_function<BackendType,powell_badly_scaled<BackendType> >("Powell-Badly-Scaled",options);
    //res |= test_function<BackendType,rosenbrock<2,BackendType> >("Rosenbrock",options);
    return res;
}

#endif
