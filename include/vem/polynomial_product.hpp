#pragma once
#include <Eigen/Sparse>

#include "vem/mesh.hpp"
namespace vem::polynomials {
namespace two {
    // the degree of the product space that multiplying two polynomials produces
    int polynomial_product_degree(int nax_degreea, int max_degreeb) {
        return a + b;
    }

    // the entries returned are the results of integrating two polynomials
    Eigen::ArrayXXd polynomial_product(const VEMMesh2 &vem, int cell, int max_degreea, int max_degreeb);
}// namespace two
}// namespace vem::polynomials
