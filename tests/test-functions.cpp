/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_TEST_COMMON_HPP_
#define UMINTL_TEST_COMMON_HPP_

#include <cstdlib>
#include <iostream>

#include "umintl/backends/cblas.hpp"
#include "umintl/minimize.hpp"
#include "umintl/optimization_context.hpp"
#include "umintl/debug.hpp"
#include "umintl/mghfuns/beale.hpp"
#include "umintl/mghfuns/powell_badly_scaled.hpp"
#include "umintl/mghfuns/brown_badly_scaled.hpp"
#include "umintl/mghfuns/rosenbrock.hpp"
#include "umintl/mghfuns/helical_valley.hpp"
#include "umintl/mghfuns/biggs_exp6.hpp"
#include "umintl/mghfuns/watson.hpp"
#include "umintl/mghfuns/variably_dimensioned.hpp"
#include "umintl/mghfuns/box_3d.hpp"
#include "umintl/mghfuns/gaussian.hpp"
#include "umintl/mghfuns/penalty1.hpp"
#include "umintl/mghfuns/penalty2.hpp"
#include "umintl/mghfuns/gulf.hpp"
#include "umintl/mghfuns/powell_singular.hpp"
#include "umintl/mghfuns/wood.hpp"
#include "umintl/mghfuns/trigonometric.hpp"
#include "umintl/mghfuns/brown_dennis.hpp"
#include "umintl/mghfuns/gaussian.hpp"

using namespace umintl;

template<class ScalarType>
struct get_backend{
    typedef umintl::backend::cblas_types<ScalarType> type;
};

template<class FunctionType>
int test_function(FunctionType const & fun, double eps=1e-6)
{
    typedef typename FunctionType::BackendType BackendType;
    typedef typename BackendType::VectorType VectorType;
    std::size_t N = fun.N();
    VectorType X0 = BackendType::create_vector(N);
    for(std::size_t i = 0 ; i < N ; ++i){
        X0[i] = (double)rand()/RAND_MAX - 0.5;
    }
    fun.init(X0);
    std::cout << "- Testing " << fun.name() << "..." << std::flush;
    double diff = umintl::check_grad<BackendType>(fun,X0,N,eps);
    if(diff>1e-5){
        std::cout << " Fail ! Diff = " << diff << "." << std::endl;
        return EXIT_FAILURE;
    }
    else
        std::cout << std::endl;
    return EXIT_SUCCESS;
}


int main(){
    typedef double ScalarType;
    typedef umintl::backend::cblas_types<ScalarType> BackendType;
    int res = EXIT_SUCCESS;
//    res |= test_function(beale<BackendType>());
//    res |= test_function(rosenbrock<BackendType>(2));
//    res |= test_function(powell_badly_scaled<BackendType>());
//    //This function is too extreme in terms of round-off error for numerical differenciation
//    //res |= test_function(brown_badly_scaled<BackendType>());
//    res |= test_function(helical_valley<BackendType>());
//    res |= test_function(rosenbrock<BackendType>(80));
    res |= test_function(watson<BackendType>(6));
    res |= test_function(box_3d<BackendType>());
    res |= test_function(variably_dimensioned<BackendType>(20));
    res |= test_function(biggs_exp6<BackendType>());
    res |= test_function(gaussian<BackendType>());
    res |= test_function(gulf<BackendType>(10));
    res |= test_function(brown_dennis<BackendType>(),1e-3);
    res |= test_function(powell_singular<BackendType>(4));
    res |= test_function(powell_singular<BackendType>(20));
    res |= test_function(wood<BackendType>());
    res |= test_function(trigonometric<BackendType>(20));
    res |= test_function(penalty1<BackendType>(10),5e-2);
    res |= test_function(penalty2<BackendType>(10));
    res |= test_function(gaussian<BackendType>());


    //res |= test_function(penalty1<BackendType>(),options);
    //res |= test_function(penalty2<BackendType>(),options);
    //res |= test_function(brown_dennis<BackendType>(),options);
    //res |= test_function(gulf<BackendType>(),options);
    //res |= test_function(trigonometric<BackendType>(),options);
    //res |= test_function(powell_singular<BackendType>(2),options);
    //res |= test_function(wood<BackendType>(),options);

    return res;
}

#endif
