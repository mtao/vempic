#include "vem/polynomial_gradient.hpp"

#include <spdlog/spdlog.h>

#include <iostream>
#include <mtao/eigen/mat_to_triplets.hpp>
#include <mtao/iterator/enumerate.hpp>

#include "vem/polynomial_partial_derivative.hpp"
#include "vem/polynomial_utils.hpp"

namespace vem::polynomials {
namespace {
template <int D, typename Func>
Eigen::SparseMatrix<double> gradient(Func &&partial_deriv_func,
                                     size_t max_degree, size_t size) {
    Eigen::SparseMatrix<double> R(D * size, size);
    std::vector<Eigen::Triplet<double>> trips =
        mtao::eigen::mat_to_triplets(partial_deriv_func(max_degree, 0));
    trips.reserve(D * trips.size());

    for (int d = 1; d < D; ++d) {
        auto dt =
            mtao::eigen::mat_to_triplets(partial_deriv_func(max_degree, d));
        int off = d * size;
        std::transform(dt.begin(), dt.end(), dt.begin(),
                       [off](const Eigen::Triplet<double> &t) {
                           return Eigen::Triplet<double>{t.row() + off, t.col(),
                                                         t.value()};
                       });
        trips.insert(trips.end(), dt.begin(), dt.end());
    }
    R.setFromTriplets(trips.begin(), trips.end());
    return R;
}
template <int D, typename Func>
Eigen::SparseMatrix<double> laplacian(Func &&partial_deriv_func,
                                      size_t max_degree, size_t size,
                                      size_t size2) {
    Eigen::SparseMatrix<double> R(size2, size);
    auto S = partial_deriv_func(max_degree, 0);
    std::vector<Eigen::Triplet<double>> trips =
        mtao::eigen::mat_to_triplets((S * S).eval());
    trips.reserve(D * trips.size());

    for (int d = 1; d < D; ++d) {
        auto S = partial_deriv_func(max_degree, d);
        auto dt = mtao::eigen::mat_to_triplets((S * S).eval());
        trips.insert(trips.end(), dt.begin(), dt.end());
    }
    R.setFromTriplets(trips.begin(), trips.end());
    return R;
}

}  // namespace
namespace two {
Eigen::SparseMatrix<double> gradient(size_t max_degree) {
    size_t size = num_monomials_upto(max_degree);
    return vem::polynomials::gradient<2>(partial_derivative, max_degree, size);
}
Eigen::SparseMatrix<double> laplacian(size_t max_degree) {
    size_t size = num_monomials_upto(max_degree);
    size_t size2 = num_monomials_upto(max_degree - 2);
    return vem::polynomials::laplacian<2>(partial_derivative, max_degree, size,
                                          size2);
}
std::array<std::tuple<double, int>, 2> gradient_single_index(size_t index) {
    std::array<std::tuple<double, int>, 2> R{
        {partial_derivative_single_index(index, 0),
         partial_derivative_single_index(index, 1)}};
    return R;
}
std::vector<std::array<std::tuple<double, int>, 2>> gradients_as_tuples(
    size_t max_degree) {
    size_t size = num_monomials_upto(max_degree);

    std::vector<std::array<std::tuple<double, int>, 2>> ret(size);
    for (auto &&[ind, ret] : mtao::iterator::enumerate(ret)) {
        ret = gradient_single_index(ind);
    }
    return ret;
}
}  // namespace two
namespace three {
Eigen::SparseMatrix<double> gradient(size_t max_degree) {
    size_t size = num_monomials_upto(max_degree);
    return vem::polynomials::gradient<3>(partial_derivative, max_degree, size);
}
Eigen::SparseMatrix<double> laplacian(size_t max_degree) {
    size_t size = num_monomials_upto(max_degree);
    size_t size2 = num_monomials_upto(max_degree - 2);
    return vem::polynomials::laplacian<3>(partial_derivative, max_degree, size,
                                          size2);
}
std::array<std::tuple<double, int>, 3> gradient_single_index(size_t index) {
    std::array<std::tuple<double, int>, 3> R{{
        partial_derivative_single_index(index, 0),
        partial_derivative_single_index(index, 1),
        partial_derivative_single_index(index, 2),
    }};
    return R;
}
std::vector<std::array<std::tuple<double, int>, 3>> gradients_as_tuples(
    size_t max_degree) {
    size_t size = num_monomials_upto(max_degree);

    std::vector<std::array<std::tuple<double, int>, 3>> ret(size);
    for (auto &&[ind, ret] : mtao::iterator::enumerate(ret)) {
        ret = gradient_single_index(ind);
    }
    return ret;
}
}  // namespace three
}  // namespace vem::polynomials
