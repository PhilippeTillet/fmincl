/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include "umintl/minimize.hpp"

class custom_stop : public umintl::stopping_criterion{
public:
    bool operator()(umintl::optimization_context & context){
        atidlas::array const & X = context.x(); //Obtain current iterate
        double X0 = X[0];
        return std::abs(X0 - 1)<1e-6;
    }
};

class rosenbrock{
public:
    rosenbrock(std::size_t N) : N_(N){ }

    void operator()(atidlas::array const & x, double& val, atidlas::array& grad, umintl::value_gradient) {
        double res=0;
        for(unsigned int i=0 ; i<N_-1;++i){
            double xi = x[i];
            double xip1 = x[i+1];
            res = res + 100*(std::pow(xip1 - xi*xi,2)) + std::pow(1 - xi,2);
        }
        val=res;

        double x0 = x[0];
        double x1 = x[1];
        grad[0] = -400*x0*(x1 - pow(x0,2)) - 2*(1 - x0);
        for(unsigned int i=1 ; i<N_-1 ; ++i){
            double xi = x[i];
            double xim1 = x[i-1];
            double xip1 = x[i+1];
            grad[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
        }
        grad[N_-1] = 200*(x[N_-1]-x[N_-2]*x[N_-2]);
    }
private:
    std::size_t N_;
};

void print_solution(umintl::optimization_result const & result, atidlas::array const& S, std::size_t D)
{
    std::cout << "Optimization complete ! " << std::endl;
    std::cout << "Solution : " << std::endl;
    std::cout << S << std::endl;
    std::cout << std::endl;
    std::cout << "Solution's value : " << result.f << std::endl;
    std::cout << "Found in " << result.iteration << " iterations / " << result.n_functions_eval << " functions eval" << " / " << result.n_gradient_eval << " gradient eval " << std::endl;
    std::cout << std::endl;
}

int main(){
    srand(0);

    unsigned int D = 10;
    atidlas::numeric_type dtype = atidlas::FLOAT_TYPE;

    std::cout << "====================" << std::endl;
    std::cout << "Minimization of the generalized Rosenbrock function" << std::endl;
    std::cout << "Dimension : " << D << std::endl;
    std::cout << "====================" << std::endl;

    atidlas::array X0(D, dtype);
    atidlas::array S(D, dtype);

    for(std::size_t i = 0 ; i < D ; ++i)
        X0[i] = 0;

    std::cout << "Starting at : " << X0 << std::endl;

    umintl::minimizer minimizer;
    rosenbrock objective(D);
    minimizer.max_iter = 100000;
    minimizer.verbosity_level=2;
    umintl::optimization_result result;

    std::cout << std::endl;

    std::cout << "--------------------" << std::endl;
    std::cout << "Steepest descent" << std::endl;
    std::cout << "--------------------" << std::endl;
    minimizer.direction = new umintl::steepest_descent();
    result = minimizer(S,objective,X0,D);
    print_solution(result,S,D);

//    std::cout << "--------------------" << std::endl;
//    std::cout << "CG [ beta = polak-ribiere , no restart ]" << std::endl;
//    std::cout << "--------------------" << std::endl;
//    minimizer.direction = new umintl::conjugate_gradient(umintl::tag::conjugate_gradient::UPDATE_POLAK_RIBIERE, umintl::tag::conjugate_gradient::NO_RESTART);
//    result = minimizer(S,objective,X0,D);
//    print_solution(result,S,D);

//    std::cout << "--------------------" << std::endl;
//    std::cout << "BFGS" << std::endl;
//    std::cout << "--------------------" << std::endl;
//    minimizer.direction = new umintl::quasi_newton();
//    result = minimizer(S,objective,X0,D);
//    print_solution(result,S,D);

//    std::cout << "--------------------" << std::endl;
//    std::cout << "L-BFGS [ memory = 5 ]" << std::endl;
//    std::cout << "--------------------" << std::endl;
//    minimizer.direction = new umintl::low_memory_quasi_newton(8);
//    result = minimizer(S,objective,X0,D);
//    print_solution(result,S,D);

//    std::cout << "--------------------" << std::endl;
//    std::cout << "Truncated Newton" << std::endl;
//    std::cout << "--------------------" << std::endl;
//    minimizer.direction = new umintl::truncated_newton();
//    result = minimizer(S,objective,X0,D);
//    print_solution(result,S,D);

//    std::cout << std::endl;
//    std::cout << "--------------------" << std::endl;
//    std::cout << "Truncated Newton" << std::endl;
//    std::cout << "Custom Stopping criterion:" << std::endl;
//    std::cout << "Stops when the first dimension is close enough to optimal" << std::endl;
//    std::cout << "--------------------" << std::endl;
//    minimizer.stopping_criterion = new custom_stop();
//    result = minimizer(S,objective,X0,D);
//    print_solution(result,S,D);



    return EXIT_SUCCESS;
}
