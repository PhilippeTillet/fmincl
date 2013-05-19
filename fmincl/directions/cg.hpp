#ifndef FMINCL_DIRECTIONS_CG_HPP_
#define FMINCL_DIRECTIONS_CG_HPP_

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

namespace fmincl{

    namespace direction{

        namespace tags{

            struct polak_ribiere{
                viennacl::scalar<double> operator()(viennacl::vector<double> const & gk
                                                    , viennacl::vector<double> const & gkm1){
                    return viennacl::linalg::inner_prod(gk,  gk - gkm1)/viennacl::linalg::inner_prod(gkm1,gkm1);
                }
            };

            struct no_restart{
                bool operator()(){
                    return false;
                }
            };

        }

        template<class BETA_POLICY = tags::polak_ribiere, class RESTART_POLICY = tags::no_restart>
        class cg{
            public:
              void operator()(viennacl::vector<double> & pk, viennacl::vector<double> const & gk, viennacl::vector<double> const * gkm1 = NULL){
                  if(gkm1==NULL || restart_()){
                      pk = -gk;
                  }
                  else{
                    viennacl::scalar<double> beta = compute_beta_(gk, *gkm1);
                    pk = -gk + beta*pk;
                  }
              }
            private:
              BETA_POLICY compute_beta_;
              RESTART_POLICY restart_;
        };

    }

}

#endif
