#ifndef FMINCL_LINE_SEARCH_TERMINATION_HPP_
#define FMINCL_LINE_SEARCH_TERMINATION_HPP_

#include "utils.hpp"
#include <cmath>

namespace fmincl{

    namespace line_search{

        class strong_wolf_powell{
        public:
            strong_wolf_powell(double const & phi_0, double const & dphi_0, double c1 = 1e-4, double c2 = 0.1) : phi_0_(phi_0), dphi_0_(dphi_0), c1_(c1), c2_(c2){ }
            bool sufficient_decrease(double ai, double phi_ai) const {
                return phi_ai <= (phi_0_ + c1_*ai*dphi_0_);
            }
            bool curvature(double dphi_ai) const{
                return std::abs(dphi_ai) <= c2_*std::abs(dphi_0_);
            }
        private:
            double const & phi_0_;
            double const & dphi_0_;
            double c1_;
            double c2_;
        };


    }
}

#endif
