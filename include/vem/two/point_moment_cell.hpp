#pragma once

#include <Eigen/Sparse>

#include "vem/cell.hpp"
#include "vem/mesh.hpp"
#include "vem/moment_basis_indexer.hpp"
#include "vem/point_sample_indexer.hpp"
namespace vem {
class PointMomentIndexer;
class PointMomentVEM2Cell : public VEM2Cell {
   public:
    using VEM2Cell::monomial_dirichlet_grammian;
    using VEM2Cell::monomial_l2_grammian;
    PointMomentVEM2Cell(const PointMomentIndexer& mom, size_t index);

    // maps each monomial to the equivalent samples
    mtao::MatXd monomial_evaluation() const;  // Hitchhiker: D

    // sample -> poly
    mtao::MatXd l2_projector() const;  // Hitchiker: \Pi^0_* = H^{-1} C
    mtao::MatXd dirichlet_projector()
        const;  // Hitchhiker: \Pi^\nabla_* = G^{-1} B

    // sample -> sample but in the span
    mtao::MatXd l2_sample_projector() const;  // Hitchiker: \Pi^0 = D\Pi^0_*
    mtao::MatXd dirichlet_sample_projector()
        const;  // Hitchhiker: \Pi^\nabla = D\Pi^\nabla_*

    mtao::MatXd monomial_l2_grammian() const;         // Hitchhiker: H
    mtao::MatXd monomial_dirichlet_grammian() const;  // Hitchihiker: \tilde G
    mtao::MatXd regularized_monomial_dirichlet_grammian()
        const;  // Hitchhiker: G

    // rows are samples, cols are monomials
    mtao::MatXd sample_monomial_l2_grammian() const;         // Hitchhiker: C
    mtao::MatXd sample_monomial_dirichlet_grammian() const;  // Hitchhiker: B

    mtao::MatXd l2_projector_error() const;
    mtao::MatXd dirichlet_projector_error() const;

    Eigen::SparseMatrix<double> local_monomial_gradient() const;

    // the number of DOFs in the whole problem
    size_t local_sample_size() const;
    // the number of DOFs in the local system
    size_t sample_size() const;

    size_t edge_interior_sample_count() const;
    size_t point_sample_size() const;
    size_t moment_degree() const;
    size_t moment_size() const;
    size_t monomial_degree() const;
    size_t monomial_size() const;

    size_t local_moment_index_offset() const;
    size_t global_moment_index_offset() const;
    size_t moment_only_global_moment_index_offset() const;
    size_t global_monomial_index_offset() const;

    // auto center() const { return mesh.C.col(index); }
    // indices of all of hte point samples
    std::vector<size_t> point_sample_indices() const;
    std::vector<size_t> sample_indices() const;
    // the sum of the lengths of all edges
    // double boundary_area() const;
    // returns the length of a single edge
    // double edge_length(size_t edge_index) const;
    // const std::map<int, bool>& edges() const;
    // returns the length of each edge according to its index
    // std::map<size_t, double> edge_lengths() const;
    // std::map<size_t, std::array<double, 2>> edge_normals() const;

    // template <typename Derived>
    // double evaluate_monomial(size_t mono_index,
    //                         const Eigen::MatrixBase<Derived>& p) const;
    template <typename Derived>
    mtao::VecXd evaluate_monomials(const Eigen::MatrixBase<Derived>& p) const;

    // same as evaluate monomials, but up to moment degree
    template <typename Derived>
    mtao::VecXd evaluate_monomials_for_moments(
        const Eigen::MatrixBase<Derived>& p) const;

    template <typename Derived, typename CDerived>
    double evaluate_monomial_function_from_block(
        const Eigen::MatrixBase<Derived>& p,
        const Eigen::MatrixBase<CDerived>& coeffs) const;

    std::map<size_t, size_t> world_to_local_point_sample_indices() const;
    std::map<size_t, size_t> world_to_local_sample_indices() const;
    std::vector<size_t> local_to_world_point_sample_indices() const;
    std::vector<size_t> local_to_world_sample_indices() const;
    mtao::iterator::detail::range_container<size_t>
    local_to_world_monomial_indices() const;
    mtao::iterator::detail::range_container<size_t> moment_indices() const;
    Eigen::SparseMatrix<double> local_to_world_sample_map() const;
    Eigen::SparseMatrix<double> local_to_world_monomial_map() const;

    template <typename Derived>
    mtao::VecXd monomial_values(const Eigen::MatrixBase<Derived>& p) const;

    const MonomialBasisIndexer& monomial_indexer() const;

    const MomentBasisIndexer& moment_indexer() const;
    const RKHSBasisIndexer& point_sample_indexer() const;

   private:
    const PointMomentIndexer* _indexer;
};

template <typename Derived>
mtao::VecXd PointMomentVEM2Cell::monomial_values(
    const Eigen::MatrixBase<Derived>& p) const {
    mtao::VecXd ret(monomial_size());
    for (int j = 0; j < ret.size(); ++j) {
        ret(j) = evaluate_monomial(j, p);
    }
    return ret;
}

template <typename Derived>
mtao::VecXd PointMomentVEM2Cell::evaluate_monomials(
    const Eigen::MatrixBase<Derived>& p) const {
    return VEM2Cell::evaluate_monomials(monomial_degree(), p);
}

template <typename Derived>
mtao::VecXd PointMomentVEM2Cell::evaluate_monomials_for_moments(
    const Eigen::MatrixBase<Derived>& p) const {
    return VEM2Cell::evaluate_monomials(moment_degree(), p);
}

template <typename Derived, typename CDerived>
double PointMomentVEM2Cell::evaluate_monomial_function_from_block(
    const Eigen::MatrixBase<Derived>& p,
    const Eigen::MatrixBase<CDerived>& coeffs) const {
    auto [start, end] = monomial_indexer().coefficient_range(cell_index());
    auto C = coeffs.segment(start, end - start);
    return VEM2Cell::evaluate_monomial_function(p, C);
}
}  // namespace vem
