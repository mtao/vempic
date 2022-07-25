#pragma once
#include <Eigen/Sparse>
#include <mtao/eigen/stl2eigen.hpp>
#include <vem/mesh.hpp>

#include "vem/polynomial_utils.hpp"

namespace vem {

class VEM3Cell {
   public:
    VEM3Cell(const VEMMesh3& mesh, const size_t cell_index);
    VEM3Cell(const VEMMesh3& mesh, const size_t cell_index, double diameter);

    const VEMMesh3& mesh() const;
    size_t cell_index() const;
    std::optional<int> cell_category() const;
    double diameter() const;
    template <typename D>
    bool is_inside(const Eigen::MatrixBase<D>& v) const;

    mtao::MatXd monomial_l2_grammian(size_t max_degree) const;
    mtao::MatXd monomial_dirichlet_grammian(size_t max_degree) const;
    // rows are cell monomials, cols are face monomials
    mtao::MatXd monomial_l2_face_grammian(size_t face_index, size_t row_degree,
                                          size_t col_degree) const;
    size_t face_count() const;
    auto center() const { return _mesh.C.col(_cell_index); }
    // the sum of the lengths of all faces
    double boundary_area() const;
    // returns the length of a single face; take a world space index
    double surface_area(size_t face_index) const;
    const std::map<int, bool>& faces() const;
    // returns the length of each face according to its index
    std::map<size_t, double> surface_areas() const;
    std::map<size_t, std::array<double, 3>> face_normals() const;

    // integrate each monomial in each domain
    mtao::VecXd monomial_integrals(size_t max_degree) const;
    mtao::VecXd face_monomial_integrals(size_t face_index,
                                        size_t max_degree) const;

    Eigen::AlignedBox<double, 3> bounding_box() const;
    //
    // integrals of the monomials along the entire face
    mtao::VecXd monomial_face_integrals(size_t max_degree,
                                        size_t face_index) const;

    // face normal with its sign turned approrpiately for this cell to face
    // outward
    mtao::Vec3d face_normal(size_t face_index) const;
    const mtao::Matrix<double, 3, 2>& face_frame(size_t face_index) const;
    mtao::Vec3d face_center(size_t face_index) const;

    // integrate each monomial along the entire boundary
    mtao::VecXd monomial_boundary_integrals(size_t max_degree) const;

    std::function<double(const mtao::Vec3d&)> monomial(size_t index) const;

    std::set<size_t> vertices() const;
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

    std::function<mtao::Vec3d(const mtao::Vec3d&)> monomial_gradient(
        size_t index) const;

    mtao::VecXd project_monomial_to_boundary(size_t face_index,
                                             size_t cell_monomial_index) const;

   private:
    const VEMMesh3& _mesh;
    const size_t _cell_index;
    const double _diameter;
};

template <typename Derived>
double VEM3Cell::evaluate_monomial(size_t mono_index,
                                   const Eigen::MatrixBase<Derived>& p) const {
    if (mono_index == 0) {
        return 1;
    }
    auto C = center();
    auto E = polynomials::three::index_to_exponents(mono_index);
    auto EE = mtao::eigen::stl2eigen(E)
                  .array()
                  .template cast<typename Derived::Scalar>();
    return ((p - C) / diameter()).array().pow(EE).prod();
}

template <typename Derived>
mtao::VecXd VEM3Cell::evaluate_monomials_by_size(
    size_t size, const Eigen::MatrixBase<Derived>& p) const {
    mtao::VecXd R(size);
    for (int j = 0; j < R.size(); ++j) {
        R(j) = evaluate_monomial(j, p);
    }
    return R;
}
template <typename Derived>
mtao::VecXd VEM3Cell::evaluate_monomials(
    size_t max_degree, const Eigen::MatrixBase<Derived>& p) const {
    size_t monomial_size = polynomials::three::num_monomials_upto(max_degree);
    return evaluate_monomials_by_size(monomial_size, p);
}
template <typename Derived, typename CDerived>
double VEM3Cell::evaluate_monomial_function(
    const Eigen::MatrixBase<Derived>& p,
    const Eigen::MatrixBase<CDerived>& coeffs) const {
    return coeffs.dot(evaluate_monomials_by_size(coeffs.size(), p));
}
template <typename A>
bool VEM3Cell::is_inside(const Eigen::MatrixBase<A>& v) const {
    return mesh().in_cell(v, cell_index());
}

template <typename D, typename B>
mtao::VecXd VEM3Cell::unweighted_least_squares_coefficients(
    size_t max_degree, const Eigen::MatrixBase<D>& P,
    const Eigen::MatrixBase<B>& values) const {
    size_t monomial_size = polynomials::three::num_monomials_upto(max_degree);

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
