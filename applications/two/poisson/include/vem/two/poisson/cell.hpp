#pragma once
#include <Eigen/Sparse>
#include <vem/two/cell.hpp>
#include <vem/two/mesh.hpp>
#include <vem/two/moment_basis_indexer.hpp>
#include <vem/two/monomial_basis_indexer.hpp>
#include <vem/two/point_sample_indexer.hpp>

namespace vem::two::poisson {

class PoissonVEM2Cell : public VEM2Cell {
   public:
    void nonlinear_warning() const;
    // mtao::MatXd KE() const;
    mtao::MatXd KEH() const;

    // C^*Pis0 + |E|(1-Pi0)^*(1-Pi0)
    mtao::MatXd M() const;
    // KEH = Grad * Div
    // Grad = M * Div^*
    mtao::MatXd Grad() const;
    mtao::MatXd CoGrad() const;

    // codiv with that takes monomials in and outputs samples
    mtao::MatXd CoGrad_mIn() const;
    // div that takes points in and generates monomials
    mtao::MatXd Grad_mOut() const;
    mtao::MatXd Pi() const;
    // \Pi^\nabla_*, Note that  \Pi = D * \Pis
    mtao::MatXd Pis() const;

    mtao::MatXd Pis0() const;
    mtao::MatXd Pi0() const;
    mtao::MatXd G() const;  // G_{qq}
    mtao::MatXd B() const;  // G_{qp}
    mtao::MatXd D() const;  // K_{pq}
    mtao::MatXd C() const;  // \int m,f
    mtao::MatXd H() const;  // (m,m)
    mtao::MatXd PErr() const;
    mtao::MatXd monomial_l2_grammian() const;
    mtao::MatXd monomial_grammian() const;
    Eigen::SparseMatrix<double> local_monomial_gradient() const;

    // the number of DOFs in the whole problem
    int system_size() const;
    // the number of DOFs in the local system
    int local_system_size() const;
    PoissonVEM2Cell(const VEMMesh2& mesh, size_t index,
                    const PointSampleIndexer& a, const MonomialBasisIndexer& b,
                    const MomentBasisIndexer& c);
    PoissonVEM2Cell(const VEM2Cell& cell, const PointSampleIndexer& a,
                    const MonomialBasisIndexer& b, const MomentBasisIndexer& c);

    // size_t edge_count() const;
    size_t edge_interior_sample_count() const;
    size_t vertex_count() const;
    size_t boundary_sample_count() const;
    size_t moment_degree() const;
    size_t moment_size() const;
    size_t monomial_degree() const;
    size_t monomial_size() const;
    // auto center() const { return mesh.C.col(index); }
    // indices of all of hte point samples
    std::vector<size_t> point_indices() const;
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

    std::map<size_t, size_t> world_to_local_point_indices() const;
    std::map<size_t, size_t> world_to_local_sample_indices() const;
    std::vector<size_t> local_to_world_point_indices() const;
    std::vector<size_t> local_to_world_sample_indices() const;
    mtao::iterator::detail::range_container<size_t>
    local_to_world_monomial_indices() const;
    Eigen::SparseMatrix<double> local_to_world_sample_map() const;
    Eigen::SparseMatrix<double> local_to_world_monomial_map() const;

    // std::set<size_t> vertices() const;
    // double volume() const;
    // double diameter() const;
    // std::function<double(const mtao::Vec2d&)> monomial(size_t index) const;
    // std::function<mtao::Vec2d(const mtao::Vec2d&)> monomial_gradient(
    //    size_t index) const;

    size_t local_moment_index_offset() const;
    // not particular to this cell
    size_t global_moment_index_offset() const;
    size_t global_monomial_index_offset() const;

    template <typename Derived>
    mtao::VecXd monomial_values(const Eigen::MatrixBase<Derived>& p) const;

   private:
    const PointSampleIndexer& point_indexer;
    const MonomialBasisIndexer& monomial_indexer;
    const MomentBasisIndexer& moment_indexer;
};

template <typename Derived>
mtao::VecXd PoissonVEM2Cell::monomial_values(
    const Eigen::MatrixBase<Derived>& p) const {
    mtao::VecXd ret(monomial_size());
    for (int j = 0; j < ret.size(); ++j) {
        ret(j) = evaluate_monomial(j, p);
    }
    return ret;
}

template <typename Derived>
mtao::VecXd PoissonVEM2Cell::evaluate_monomials(
    const Eigen::MatrixBase<Derived>& p) const {
    return VEM2Cell::evaluate_monomials(monomial_degree(), p);
}

template <typename Derived>
mtao::VecXd PoissonVEM2Cell::evaluate_monomials_for_moments(
    const Eigen::MatrixBase<Derived>& p) const {
    return VEM2Cell::evaluate_monomials(moment_degree(), p);
}

template <typename Derived, typename CDerived>
double PoissonVEM2Cell::evaluate_monomial_function_from_block(
    const Eigen::MatrixBase<Derived>& p,
    const Eigen::MatrixBase<CDerived>& coeffs) const {
    auto [start, end] = monomial_indexer.coefficient_range(cell_index());
    auto C = coeffs.segment(start, end - start);
    return VEM2Cell::evaluate_monomial_function(p, C);
}
}  // namespace vem::poisson_2d
