/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include <iostream>

#include "test-common.hpp"

using namespace umintl;

int main(){
    srand(0);

    int result = EXIT_SUCCESS;

    typedef typename get_backend<double>::type BackendType;

    result |= test_option("Conjugate Gradient [Double - Polak-Ribière]", new conjugate_gradient<BackendType>(umintl::tag::conjugate_gradient::UPDATE_POLAK_RIBIERE));
    result |= test_option("Conjugate Gradient [Double - Gilbert-Nocedal]", new conjugate_gradient<BackendType>(umintl::tag::conjugate_gradient::UPDATE_GILBERT_NOCEDAL));

    return result;

}
