/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include "fmincl/minimize.hpp"
#include "rosenbrock.hpp"

template<class BackendType>
double test_result(std::size_t N, typename BackendType::VectorType const & vec){
    typedef typename BackendType::ScalarType ScalarType;
    ScalarType diff = 0;
    for(std::size_t i = 0 ; i < N ; ++i){
        if(std::isnan(vec[i])) return INFINITY;
        diff = std::max(diff,std::fabs(vec[i]-static_cast<ScalarType>(1)));
    }
    return diff;
}

template<class BackendType>
int run_test_impl(std::size_t dimension, typename BackendType::ScalarType epsilon){
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;

    int res = EXIT_SUCCESS;

    ScalarType diff;
    VectorType X0 = BackendType::create_vector(dimension);

    for(unsigned int i = 0 ; i < dimension ; ++i) X0[i] = 0.01*(ScalarType)rand()/RAND_MAX;

#define TEST_OPTIONS(options) \
    if((diff = test_result<BackendType>(dimension, fmincl::minimize<BackendType>(rosenbrock<BackendType>(dimension),X0,dimension,options)))>epsilon){ \
        std::cout << "## Failure! Diff = " << diff << std::endl; \
        res = EXIT_FAILURE; \
    }

    std::cout << "* Testing BFGS..." << std::endl;
    TEST_OPTIONS(fmincl::optimization_options(new fmincl::quasi_newton_tag(new fmincl::bfgs_tag())))

    std::cout << "* Testing L-BFGS [m=1] ..." << std::endl;
    TEST_OPTIONS(fmincl::optimization_options(new fmincl::quasi_newton_tag(new fmincl::lbfgs_tag(1))))
    std::cout << "* Testing L-BFGS [m=4] ..." << std::endl;
    TEST_OPTIONS(fmincl::optimization_options(new fmincl::quasi_newton_tag(new fmincl::lbfgs_tag(4))))
    std::cout << "* Testing L-BFGS [m=16] ..." << std::endl;
    TEST_OPTIONS(fmincl::optimization_options(new fmincl::quasi_newton_tag(new fmincl::lbfgs_tag(16))))

    std::cout << "* Testing Polak-Ribiere [No restart]..." << std::endl;
    TEST_OPTIONS(fmincl::optimization_options(new fmincl::cg_tag(new fmincl::polak_ribiere_tag(), new fmincl::no_restart_tag())))

    BackendType::delete_if_dynamically_allocated(X0);

    return res;
}

template<class BackendType>
int run_test(typename BackendType::ScalarType epsilon){
    srand(0);
    std::vector<std::size_t> dimensions;
    dimensions.push_back(2);
    dimensions.push_back(40);


    for(std::vector<std::size_t>::iterator it = dimensions.begin(); it != dimensions.end(); ++it){
        std::cout << std::endl;
        std::cout << "Dimension: " << *it << std::endl;
        std::cout << "-------------------" << std::endl;
        if(run_test_impl<BackendType>(*it, epsilon)==EXIT_FAILURE){
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
