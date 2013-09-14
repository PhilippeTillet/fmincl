#ifndef FMINCL_BEALE_HPP_
#define FMINCL_BEALE_HPP_

#include <cmath>

template<class BackendType>
class beale{
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
public:
    static const std::size_t N = 2;

    static ScalarType true_minimum_value() { return 0; }

    ScalarType operator()(VectorType const & V, VectorType * grad) const {
        ScalarType x=V[0], y=V[1];
        ScalarType res = std::pow(1.5   -x+x*y,2)
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
