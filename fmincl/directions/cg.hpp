#ifndef FMINCL_DIRECTIONS_CG_HPP_
#define FMINCL_DIRECTIONS_CG_HPP_

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

namespace fmincl{

    namespace direction{

        namespace tags{

            struct polak_ribiere{
                viennacl::scalar<double> operator()(viennacl::vector<double> const & gkp1
                                                    , viennacl::vector<double> const & gk){
                    return viennacl::linalg::inner_prod(gkp1,  gkp1 - gk)/viennacl::linalg::inner_prod(gk,gk);
                }
            };

            struct no_restart{
                bool operator()(){
                    return false;
                }
            };

            template<class BETA_POLICY = polak_ribiere, class RESTART_POLICY = no_restart>
            class cg{
                public:
                cg(double tol = 1e-5, unsigned int max_iter = 1000 ) : tol_ (tol), max_iter_ (max_iter), last_grad_(NULL) {}
                  double tolerance() const { return tol_; }
                  unsigned int max_iter() const { return max_iter_; }
                  void operator()(viennacl::vector<double> & pk, viennacl::vector<double> gk){
                      if(last_grad_==NULL || restart_()){
                          pk = -gk;
                      }
                      else{
                        viennacl::scalar<double> beta = compute_beta_(gk, *last_grad_);
                        pk = -gk + beta*pk;
                      }
                  }
                private:
                  double tol_;
                  unsigned int max_iter_;
                  viennacl::vector<double> * last_grad_;
                  BETA_POLICY compute_beta_;
                  RESTART_POLICY restart_;
            };
        }

    }

}
