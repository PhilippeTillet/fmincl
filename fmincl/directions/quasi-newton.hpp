#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <viennacl/matrix.hpp>
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/vector_operations.hpp>
#include <viennacl/linalg/matrix_operations.hpp>
#include "viennacl/linalg/prod.hpp"
namespace fmincl{

    namespace direction{


        class quasi_newton{
        public:
            quasi_newton(viennacl::vector<double> & pk, viennacl::vector<double> const & xk, viennacl::vector<double> const & gk) : pk_(pk), xk_(xk), gk_(gk), is_first_update_(true){

            }

            void operator()(){
                if(gkm1_.empty()){
                    pk_ = -gk_;
                }
                else{
                    viennacl::vector<double> skm1 = xk_ - xkm1_;
                    viennacl::vector<double> ykm1 = gk_ - gkm1_;

                    viennacl::scalar<double> rho = viennacl::linalg::inner_prod(skm1,ykm1);

                    if(is_first_update_=true){
                        viennacl::scalar<double> nykm1 = viennacl::linalg::inner_prod(ykm1,ykm1);
                        viennacl::scalar<double> scale = rho/nykm1;
                        Hk = viennacl::identity_matrix<double>(gk_.size());
                        Hk *= scale;
                        is_first_update_=false;
                    }

                    viennacl::scalar<double> rho2 = rho*rho;
                    viennacl::vector<double> Hkm1ykm1 = viennacl::linalg::prod(Hk,ykm1);
                    viennacl::scalar<double> ip = viennacl::linalg::inner_prod(ykm1,Hkm1ykm1);
                    viennacl::scalar<double> alpha = (rho+ip)/rho2;
                    viennacl::scalar<double> beta = viennacl::scalar<double>(1)/rho;

                    Hk += alpha*viennacl::linalg::outer_prod(skm1,skm1);
                    Hk += beta*viennacl::linalg::outer_prod(Hkm1ykm1,skm1);
                    Hk += beta*viennacl::linalg::outer_prod(skm1, Hkm1ykm1);


                    viennacl::vector<double> tmp = viennacl::linalg::prod(Hk,gk_);
                    pk_ = -tmp;
                }


                xkm1_ = xk_;
                gkm1_ = gk_;
            }

        private:
            viennacl::vector<double> & pk_;
            viennacl::vector<double> const & xk_;
            viennacl::vector<double> const & gk_;
            viennacl::vector<double> xkm1_;
            viennacl::vector<double> gkm1_;
            viennacl::matrix<double> Hk;
            bool is_first_update_;
        };

    }
}
#endif
