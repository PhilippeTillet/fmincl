#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <viennacl/matrix.hpp>
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/vector_operations.hpp>
#include <viennacl/linalg/matrix_operations.hpp>
#include "viennacl/linalg/prod.hpp"

#include "fmincl/utils.hpp"

namespace fmincl{

    namespace direction{


        class quasi_newton{
        public:
            quasi_newton() : is_first_update_(true){

            }


            void operator()(detail::state_ref & state){
                if(gkm1_.empty()){
                    state.p = -state.g;
                }
                else{
                    viennacl::vector<double> skm1 = state.x - xkm1_;
                    viennacl::vector<double> ykm1 = state.g - gkm1_;


                    if(is_first_update_==true){
                        viennacl::scalar<double> ipsy = viennacl::linalg::inner_prod(skm1,ykm1);
                        viennacl::scalar<double> nykm1 = viennacl::linalg::inner_prod(ykm1,ykm1);
                        viennacl::scalar<double> scale = ipsy/nykm1;
                        Hk = viennacl::identity_matrix<double>(state.g.size());
                        Hk *= 1;
                        is_first_update_=false;
                    }

                    viennacl::scalar<double> rho = (double)(1)/viennacl::linalg::inner_prod(skm1,ykm1);
                    viennacl::scalar<double> rho2 = rho*rho;
                    viennacl::vector<double> Hy = viennacl::linalg::prod(Hk,ykm1);
                    viennacl::scalar<double> n2y = viennacl::linalg::inner_prod(ykm1,Hy);

                    Hk -= rho*viennacl::linalg::outer_prod(Hy,skm1);
                    Hk -= rho*viennacl::linalg::outer_prod(skm1,Hy);
                    Hk += rho2*n2y*viennacl::linalg::outer_prod(skm1,skm1);
                    Hk += rho*viennacl::linalg::outer_prod(skm1,skm1);


                    viennacl::vector<double> tmp = viennacl::linalg::prod(Hk,state.g);
                    state.p = -tmp;
                }


                xkm1_ = state.x;
                gkm1_ = state.g;
            }

        private:
            viennacl::vector<double> xkm1_;
            viennacl::vector<double> gkm1_;
            viennacl::matrix<double> Hk;
            bool is_first_update_;
        };

    }
}
#endif
