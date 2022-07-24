#pragma once
#include <Eigen/Sparse>
#include <mtao/eigen/stl2eigen.hpp>

#include "vem/mesh.hpp"
#include "vem/partitioned_coefficient_indexer.hpp"
#include "vem/polynomial_utils.hpp"
#include "vem/monomial_basis_indexer_new.hpp"

namespace vem {
namespace detail {
template <int D, int E>
class MonomialBasisIndexer;
}
    using MonomialBasisIndexer = detail::MonomialBasisIndexer<2,2>;
    using MonomialBasisIndexer3 = detail::MonomialBasisIndexer<3,3>;

namespace detail {
// an indexer into coefficients that are _JUST_ monomials
// D is the dimension of the monomial, E is the dimension of the embedding
template <int D, int E = D>
class MonomialBasisIndexer : public PartitionedCoefficientIndexer {
   public:
    using MeshType = VEMMesh<E>;
    using Vec = typename mtao::Vector<double, E>;
    // pass in the number of samples are held on the interior of each edge
    MonomialBasisIndexer(MonomialBasisIndexer &&o) = default;
    MonomialBasisIndexer(const MonomialBasisIndexer &o)
        : MonomialBasisIndexer(o.mesh(), o._degrees, o._diameters) {}
    // MonomialBasisIndexer(const MonomialBasisIndexer &o) = default;
    MonomialBasisIndexer(const MeshType &mesh, size_t degree);
    MonomialBasisIndexer(const MeshType &mesh, std::vector<size_t> degrees,
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
    std::function<double(const Vec &)> monomial(size_t cell_index,
                                                size_t monomial_index) const;
    // similar for gradients
    std::function<Vec(const Vec &)> monomial_gradient(
        size_t cell_index, size_t monomial_index) const;
    template <typename Derived>
    double evaluate_monomial(size_t cell, size_t index,
                             const Eigen::MatrixBase<Derived> &P) const;

    template <typename Derived>
    Vec evaluate_monomial_gradient(size_t cell, size_t index,
                                   const Eigen::MatrixBase<Derived> &P) const;

    // integrate each monomial in each domain
    mtao::VecXd monomial_integrals(size_t cell_index) const;
    mtao::VecXd monomial_integrals(size_t cell_index, int max_degree) const;

    // integrate each monomial along the edges
    mtao::VecXd monomial_edge_integrals(size_t cell_index) const;

    double diameter(size_t index) const;
    Vec center(size_t index) const;

    [[deprecated]] std::vector<double> &cell_diameters() { return _diameters; }
    [[deprecated]] const std::vector<double> &cell_diameters() const {
        return _diameters;
    }
    // beware that after setting degrees the offset data needs
    // to be regenerated
    [[deprecated]] std::vector<size_t> &cell_degrees() { return _degrees; }
    [[deprecated]] const std::vector<size_t> &cell_degrees() const {
        return _degrees;
    }

    std::vector<double> &diameters() { return _diameters; }
    const std::vector<double> &diameters() const { return _diameters; }
    // beware that after setting degrees the offset data needs to be regenerated
    std::vector<size_t> &degrees() { return _degrees; }
    const std::vector<size_t> &degrees() const { return _degrees; }

    // returns a coeff_size -> 2*coeff_size matrix of gradient entries
    Eigen::SparseMatrix<double> gradient() const;
    Eigen::SparseMatrix<double> divergence() const;

    const MeshType &mesh() const { return _mesh; }

   private:
    const MeshType &_mesh;
    // offsets according to the numbers of monomials stored in each cell
    std::vector<size_t> _degrees;
    std::vector<double> _diameters;

   private:
    void fill_diameters();
};
template <int D, int E>
template <typename Derived>
double MonomialBasisIndexer<D, E>::evaluate_monomial(
    size_t cell, size_t index, const Eigen::MatrixBase<Derived> &P) const {
    if (index == 0) {
        return 1;
    }
    auto C = center(cell);
    auto Ex = polynomials::index_to_exponents<D>(index);
    double diameter = _diameters.at(cell);
    auto EE = mtao::eigen::stl2eigen(Ex)
                  .array()
                  .template cast<typename Derived::Scalar>();
    return ((P - C) / diameter).array().pow(EE).prod();
}
}  // namespace detail
}  // namespace vem
#include "vem/monomial_basis_indexer2.hpp"
#include "vem/monomial_basis_indexer_impl.hpp"
