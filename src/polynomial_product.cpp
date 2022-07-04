#include "vem/polynomial_product.hpp"

#include "vem/monomial_cell_integrals.hpp"
#include "vem/polynomial_utils.hpp"
#include <mtao/types.hpp>
namespace vem::polynomials {
namespace two {

    // the quadratic form induced by multiplying two polynomials with one another
    Eigen::ArrayXXd polynomial_product(const VEMMesh2 &vem, int cell_index, int max_degreea, int max_degreeb) {
        int max_degree = max_degreea + max_degreeb;
        mtao::VecXd integrals =
          monomial_cell_integrals(vem, cell_index, max_degree);
        int a_poly_count = num_monomials_upto(max_degreea);
        int b_poly_count = num_monomials_upto(max_degreeb);

        Eigen::ArrayXXd P(a_poly_count, b_poly_count);
        for (int polya = 0; polya < a_poly_count; ++polya) {
            for (int polyb = 0; polyb < b_poly_count; ++polyb) {
                auto [ax, ay] = index_to_exponents(polya);
                auto [bx, by] = index_to_exponents(polyb);

                P(polya, polyb) = exponents_to_index(ax + bx, ay + by);
            }
        }
    }
}// namespace two
}// namespace vem::polynomials
