#ifndef FMINCL_NONLINEAR_LEAST_SQUARE_HPP_
#define FMINCL_NONLINEAR_LEAST_SQUARE_HPP_

#include <cmath>
#include <vector>
#include <string>
#include <cassert>

using namespace std;

template<class _BackendType>
class sum_square{
public:
    typedef _BackendType BackendType;
private:
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
protected:
    ScalarType & get(ScalarType *A, std::size_t m, std::size_t n) const {
        return A[m*N_+n];
    }
    std::string to_string(std::size_t n) const {
        std::ostringstream oss;
        oss << n;
        return oss.str();
    }
public:
    sum_square(std::string const & name, std::size_t M, std::size_t N, ScalarType global_minimum) : name_(name), M_(M), N_(N), global_minimum_(global_minimum){ }
    std::vector<ScalarType> local_minima() const { return local_minima_; }
    ScalarType global_minimum() const { return global_minimum_; }
    std::size_t M() const { return M_; }
    std::size_t N() const { return N_; }
    std::string name() const{ return name_+" ["+to_string(N_)+"]"; }
    virtual void init(VectorType &X) const = 0;
    virtual void fill_dym_dxn(VectorType const & V, ScalarType * res) const = 0;
    virtual void fill_ym(VectorType const & V, ScalarType * res) const = 0;
    void operator()(VectorType const & V, ScalarType * val, VectorType * grad)const{
        ScalarType* y = new ScalarType[M_];
        for(std::size_t m = 0 ; m < M_ ; ++m)
            y[m] = 0;
        fill_ym(V,y);
        ScalarType* dy_dx = new ScalarType[M_*N_];
        for(std::size_t m = 0 ; m < M_ ; ++m)
            for(std::size_t n = 0 ; n < N_ ; ++n)
                get(dy_dx,m,n) = 0;
        fill_dym_dxn(V,dy_dx);
        if(val){
            ScalarType res = 0;
            for(std::size_t m = 0 ; m < M_ ; ++m)
                res += std::pow(y[m],2);
            *val = res;
        }
        if(grad){
            for(std::size_t n = 0 ; n < N_ ; ++n){
                (*grad)[n] = 0;
                for(std::size_t m = 0 ; m < M_ ; ++m){
                    (*grad)[n] += 2*y[m]*get(dy_dx,m,n);
                }
            }
        }
        delete[] dy_dx;
        delete[] y;
    }
protected:
    std::string name_;
    std::size_t M_;
    std::size_t N_;
    std::vector<double> local_minima_;
    double global_minimum_;
};

#endif
