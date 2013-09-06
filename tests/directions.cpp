/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include "fmincl/backends/eigen.hpp"
#include "fmincl/minimize.hpp"
#include "rosenbrock.hpp"
#include "Eigen/Dense"

template<class ScalarType>
double test_result(Eigen::Matrix<ScalarType,Eigen::Dynamic,1>  const & vec){
    ScalarType diff = 0;
    for(std::size_t i = 0 ; i < vec.size() ; ++i){
        if(std::isnan(vec[i])) return NAN;
        diff = std::max(diff,std::fabs(vec[i]-static_cast<ScalarType>(1)));
    }
    return diff;
}

template<class ScalarType>
int run_test(std::size_t dimension, ScalarType epsilon){
    typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> VectorType;
    typedef fmincl::backend::EigenTypes<ScalarType> BackendType;
    ScalarType diff;
    VectorType X0(dimension);
    for(unsigned int i = 0 ; i < dimension ; ++i) X0(i) = 0.01*(double)rand()/RAND_MAX;

#define TEST_OPTIONS(options) \
    if((diff = test_result(fmincl::minimize<BackendType>(rosenbrock<ScalarType>(),X0, options)))>epsilon){ \
        std::cout << "## Failure! Diff = " << diff << std::endl; \
        return EXIT_FAILURE; \
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
}

int main(){
    srand(0);
    double epsilon = 1e-5;
    std::vector<std::size_t> dimensions;
    dimensions.push_back(2);
    dimensions.push_back(10);

    std::cout << "====================" << std::endl;
    std::cout << "      Testing       "  << std::endl;
    std::cout << "     Directions     " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Float:" << std::endl;
    std::cout << "-------------------" << std::endl;
    for(std::vector<std::size_t>::iterator it = dimensions.begin(); it != dimensions.end(); ++it){
        std::cout << std::endl;
        std::cout << "Dimension: " << *it << std::endl;
        std::cout << "-------------------" << std::endl;
        if(run_test<float>(*it, epsilon)==EXIT_FAILURE){
            return EXIT_FAILURE;
        }
    }

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Double" << std::endl;
    std::cout << "-------------------" << std::endl;
    for(std::vector<std::size_t>::iterator it = dimensions.begin(); it != dimensions.end(); ++it){
        std::cout << std::endl;
        std::cout << "Dimension: " << *it << std::endl;
        std::cout << "-------------------" << std::endl;
        if(run_test<double>(*it, epsilon)==EXIT_FAILURE){
            return EXIT_FAILURE;
        }
    }

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Test passed!" << std::endl;
    return EXIT_SUCCESS;
}
