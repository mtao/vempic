#pragma once
#include <Eigen/Sparse>
#include <mtao/eigen/stl2eigen.hpp>
#include "mesh.hpp"

#include "vem/polynomials/utils.hpp"

namespace vem::two {

class VEM2Cell {
   public:
    VEM2Cell(const VEMMesh2& mesh, const size_t cell_index);
    VEM2Cell(const VEMMesh2& mesh, const size_t cell_index, double diameter);

    const VEMMesh2& mesh() const;
    size_t cell_index() const;
    double diameter() const;

    Eigen::AlignedBox<double, 2> bounding_box() const;
    template <typename D>
    bool is_inside(const Eigen::MatrixBase<D>& v) const;

    mtao::MatXd monomial_l2_grammian(size_t max_degree) const;
    // integrates monomials along the boundary
    // cells are rows, edges are cols

    mtao::MatXd monomial_l2_edge_grammian(size_t edge_index, size_t row_degree,
                                          size_t col_degree) const;

    mtao::MatXd monomial_dirichlet_grammian(size_t max_degree) const;

    size_t edge_count() const;
    auto center() const { return _mesh.C.col(_cell_index); }
    // the sum of the lengths of all edges
    double boundary_area() const;
    // returns the length of a single edge; take a world space index
    double edge_length(size_t edge_index) const;
    const std::map<int, bool>& edges() const;
    // returns the length of each edge according to its index
    std::map<size_t, double> edge_lengths() const;
    std::map<size_t, std::array<double, 2>> edge_normals() const;

    // integrate each monomial in each domain
    mtao::VecXd monomial_integrals(size_t max_degree) const;

    //
    mtao::VecXd monomial_edge_integrals(size_t max_degree,
                                        size_t edge_index) const;

    // integrate each monomial along the entire boundary
    mtao::VecXd monomial_boundary_integrals(size_t max_degree) const;

    std::function<double(const mtao::Vec2d&)> monomial(size_t index) const;
    std::function<mtao::Vec2d(const mtao::Vec2d&)> monomial_gradient(
        size_t index) const;

    std::set<size_t> vertices() const;
    size_t vertex_count() const;
    double volume() const;

    template <typename Derived>
    double evaluate_monomial(size_t mono_index,
                             const Eigen::MatrixBase<Derived>& p) const;
    template <typename Derived>
    mtao::VecXd evaluate_monomials(size_t max_degree,
                                   const Eigen::MatrixBase<Derived>& p) const;

    // specify the number of monomials rather than degree
    template <typename Derived>
    mtao::VecXd evaluate_monomials_by_size(
        size_t size, const Eigen::MatrixBase<Derived>& p) const;

    template <typename Derived, typename CDerived>
    double evaluate_monomial_function(
        const Eigen::MatrixBase<Derived>& p,
        const Eigen::MatrixBase<CDerived>& coeffs) const;
    Eigen::SparseMatrix<double> monomial_to_monomial_gradient(
        size_t max_degree) const;

    template <typename A, typename B>
    mtao::VecXd unweighted_least_squares_coefficients(
        size_t max_degree, const Eigen::MatrixBase<A>& P,
        const Eigen::MatrixBase<B>& values) const;

   private:
    const VEMMesh2& _mesh;
    const size_t _cell_index;
    const double _diameter;
};

template <typename Derived>
double VEM2Cell::evaluate_monomial(size_t mono_index,
                                   const Eigen::MatrixBase<Derived>& p) const {
    if (mono_index == 0) {
        return 1;
    }
    auto C = center();
    auto E = polynomials::two::index_to_exponents(mono_index);
    auto EE = mtao::eigen::stl2eigen(E)
                  .array()
                  .template cast<typename Derived::Scalar>();
    return ((p - C) / diameter()).array().pow(EE).prod();
}

template <typename Derived>
mtao::VecXd VEM2Cell::evaluate_monomials_by_size(
    size_t size, const Eigen::MatrixBase<Derived>& p) const {
    mtao::VecXd R(size);
    for (int j = 0; j < R.size(); ++j) {
        R(j) = evaluate_monomial(j, p);
    }
    return R;
}
template <typename Derived>
mtao::VecXd VEM2Cell::evaluate_monomials(
    size_t max_degree, const Eigen::MatrixBase<Derived>& p) const {
    size_t monomial_size = polynomials::two::num_monomials_upto(max_degree);
    return evaluate_monomials_by_size(monomial_size, p);
}
template <typename Derived, typename CDerived>
double VEM2Cell::evaluate_monomial_function(
    const Eigen::MatrixBase<Derived>& p,
    const Eigen::MatrixBase<CDerived>& coeffs) const {
    return coeffs.dot(evaluate_monomials_by_size(coeffs.size(), p));
}
template <typename A>
bool VEM2Cell::is_inside(const Eigen::MatrixBase<A>& v) const {
    return mesh().in_cell(v, cell_index());
}

template <typename D, typename B>
mtao::VecXd VEM2Cell::unweighted_least_squares_coefficients(
    size_t max_degree, const Eigen::MatrixBase<D>& P,
    const Eigen::MatrixBase<B>& values) const {
    size_t monomial_size = polynomials::two::num_monomials_upto(max_degree);

    mtao::MatXd A(monomial_size, P.cols());
    for (int j = 0; j < P.cols(); ++j) {
        A.col(j) = evaluate_monomials_by_size(monomial_size, P.col(j));
    }
    // columns are function evaluations
    // rows are in poly size
    // A.transpose() * c = values
    // A * A.transpose() * c = A * values
    return (A * A.transpose()).ldlt().solve(A * values);
}
}  // namespace vem
