#pragma once
#include <Eigen/Sparse>

namespace vem::polynomials {
namespace two {
// returns a map of the partial derivative of all monomials up to a specified
// degree along a specified axis
Eigen::SparseMatrix<double> partial_derivative(size_t max_degree, size_t axis);
// returns an optional to handle the case that the derivative is 0
std::tuple<double, int> partial_derivative_single_index(size_t index,
                                                        size_t axis);

}  // namespace two

namespace three {
// returns a map of the partial derivative of all monomials up to a specified
// degree along a specified axis
Eigen::SparseMatrix<double> partial_derivative(size_t max_degree, size_t axis);
// returns an optional to handle the case that the derivative is 0
std::tuple<double, int> partial_derivative_single_index(size_t index,
                                                        size_t axis);

}  // namespace three
}  // namespace vem::polynomials
