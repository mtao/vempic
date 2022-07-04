#pragma once
#include <Eigen/Sparse>
#include <mtao/eigen/stl2eigen.hpp>

#include "vem/mesh.hpp"
#include "vem/partitioned_coefficient_indexer.hpp"
#include "vem/polynomial_utils.hpp"
#include "vem/monomial_basis_indexer_new.hpp"

namespace vem {
    using MonomialBasisIndexer = detail::MonomialBasisIndexer<2,2>;
    using MonomialBasisIndexer3 = detail::MonomialBasisIndexer<3,3>;
    /*
// an indexer into coefficients that are _JUST_ monomials
class MonomialBasisIndexer : public PartitionedCoefficientIndexer {
   public:
    // pass in the number of samples are held on the interior of each edge
    MonomialBasisIndexer(MonomialBasisIndexer &&o) = default;
    MonomialBasisIndexer(const MonomialBasisIndexer &o)
        : MonomialBasisIndexer(o.mesh(), o._cell_degrees, o._cell_diameters) {}
    // MonomialBasisIndexer(const MonomialBasisIndexer &o) = default;
    MonomialBasisIndexer(const VEMMesh2 &mesh, size_t degree);
    MonomialBasisIndexer(const VEMMesh2 &mesh, std::vector<size_t> degrees,
                         std::vector<double> diameters = {});

    MonomialBasisIndexer relative_degree_indexer(int degree_offset) const;
    MonomialBasisIndexer derivative_indexer() const;
    MonomialBasisIndexer antiderivative_indexer() const;

    size_t num_monomials(size_t index) const;
    size_t num_coefficients(size_t index) const;
    size_t num_coefficients() const;
    size_t degree(size_t index) const;
    size_t coefficient_offset(size_t index) const;
    std::array<size_t, 2> coefficient_range(size_t index) const;

    // given a cell and a monomial index (no offsetting!) get the polynomial
    // NOTE that this works for arbitrarily high monomial indices
    std::function<double(const mtao::Vec2d &)> monomial(
        size_t cell_index, size_t monomial_index) const;
    // similar for gradients
    std::function<mtao::Vec2d(const mtao::Vec2d &)> monomial_gradient(
        size_t cell_index, size_t monomial_index) const;
    template <typename Derived>
    double evaluate_monomial(size_t cell, size_t index,
                             const Eigen::MatrixBase<Derived> &P) const;

    template <typename Derived>
    mtao::Vec2d evaluate_monomial_gradient(
        size_t cell, size_t index, const Eigen::MatrixBase<Derived> &P) const;

    // integrate each monomial in each domain
    mtao::VecXd monomial_integrals(size_t cell_index) const;
    mtao::VecXd monomial_integrals(size_t cell_index, int max_degree) const;

    // integrate each monomial along the edges
    mtao::VecXd monomial_edge_integrals(size_t cell_index) const;

    double diameter(size_t index) const;

    std::vector<double> &cell_diameters() { return _cell_diameters; }
    const std::vector<double> &cell_diameters() const {
        return _cell_diameters;
    }
    // beware that after setting degrees the offset data needs to be regenerated
    std::vector<size_t> &cell_degrees() { return _cell_degrees; }
    const std::vector<size_t> &cell_degrees() const { return _cell_degrees; }

    // returns a coeff_size -> 2*coeff_size matrix of gradient entries
    Eigen::SparseMatrix<double> gradient() const;
    Eigen::SparseMatrix<double> divergence() const;

    const VEMMesh2 &mesh() const { return _mesh; }

   private:
    const VEMMesh2 &_mesh;
    // offsets according to the numbers of monomials stored in each cell
    std::vector<size_t> _cell_degrees;
    std::vector<double> _cell_diameters;

   private:
    void fill_diameters();
};

template <typename Derived>
double MonomialBasisIndexer::evaluate_monomial(
    size_t cell, size_t index, const Eigen::MatrixBase<Derived> &P) const {
    if (index == 0) {
        return 1;
    }
    auto C = _mesh.C.col(cell);
    auto E = polynomials::two::index_to_exponents(index);
    double diameter = _cell_diameters.at(cell);
    auto EE = mtao::eigen::stl2eigen(E)
                  .array()
                  .template cast<typename Derived::Scalar>();
    return ((P - C) / diameter).array().pow(EE).prod();
}
*/
}  // namespace vem
