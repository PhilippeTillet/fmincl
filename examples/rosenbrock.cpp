/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>

#include "umintl/backends/cblas.hpp"
#include "umintl/minimize.hpp"
#include "umintl/mghfuns/powell_singular.hpp"

typedef double ScalarType;
typedef ScalarType* VectorType;
typedef umintl::backend::cblas_types<ScalarType> BackendType;

class rosenbrock
{
public:
    rosenbrock(std::size_t N) : N_(N){ }

    void operator()(ScalarType* const & x, ScalarType& val, VectorType & grad, umintl::value_gradient) {
        ScalarType res=0;
        for(unsigned int i=0 ; i<N_-1;++i){
            res = res + 100*(pow(x[i+1] - x[i]*x[i],2)) + pow(1 - x[i],2);
        }
        val=res;

        grad[0] = -400*x[0]*(x[1] - pow(x[0],2)) - 2*(1 - x[0]);
        for(unsigned int i=1 ; i<N_-1 ; ++i){
            ScalarType xi = x[i];
            ScalarType xim1 = x[i-1];
            ScalarType xip1 = x[i+1];
            grad[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
        }
        grad[N_-1] = 200*(x[N_-1]-x[N_-2]*x[N_-2]);
    }
private:
    std::size_t N_;
};

int main()
{
    srand(0);

    unsigned int D = 100;
    rosenbrock objective(D);

    ScalarType* X0 = new ScalarType[D];
    ScalarType* S = new ScalarType[D];
    for(std::size_t i = 0 ; i < D ; ++i)
        X0[i] = 2*(float)rand()/RAND_MAX - 1;

    umintl::minimizer<BackendType> minimizer;
    minimizer.max_iter = 10000;
    minimizer.verbosity_level = 2;
//    minimizer.direction = new umintl::low_memory_quasi_newton<BackendType>(5);
    minimizer.direction = new umintl::quasi_newton<BackendType>();
//    minimizer.direction = new umintl::conjugate_gradient<BackendType>();
    minimizer(S,objective,X0,D);

    return EXIT_SUCCESS;
}
