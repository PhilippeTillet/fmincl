#ifndef FMINCL_BEALE_HPP_
#define FMINCL_BEALE_HPP_

#include <cmath>
#include <vector>

template<class BackendType>
class beale{
    typedef typename BackendType::VectorType VectorType;
public:
    static const std::size_t N = 2;

    static double true_minimum_value() { return 0; }

    static void local_minima_value(std::vector<double> &) { }

    static void init(VectorType & X){ X[0] = 1; X[1] = 1; }

    double operator()(VectorType const & V, VectorType * grad) const {
        double x=V[0], y=V[1];
        double res = std::pow(1.5   -x+x*y,2)
                        +std::pow(2.25  -x+x*y*y,2)
                        +std::pow(2.625 -x+x*y*y*y,2);
        if(grad){
         (*grad)[0] = 2*(-1+y)*(1.5 -x+x*y)
                    + 2*(-1+y*y)*(2.25 -x+x*y*y)
                    + 2*(-1+y*y)*(2.625 -x+x*y*y*y);

         (*grad)[1] = 2*(x)*(1.5 -x+x*y)
                    + 2*(2*x*y)*(2.25 -x+x*y*y)
                    + 2*(3*x*y*y)*(2.625 -x+x*y*y*y);
        }
        return res;
    }

};

#endif
