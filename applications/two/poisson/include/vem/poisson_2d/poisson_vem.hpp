#pragma once
#include <Eigen/Sparse>
#include <vem/mesh.hpp>
#include <vem/moment_basis_indexer.hpp>
#include <vem/monomial_basis_indexer.hpp>
#include <vem/point_sample_indexer.hpp>

#include "vem/poisson_2d/constraints.hpp"
#include "vem/poisson_2d/poisson_vem_cell.hpp"

namespace vem::poisson_2d {
struct PoissonVEM2 {
    enum class CellWeightWeightMode : char { Unweighted, AreaWeighted };
    PoissonVEM2Cell get_cell(size_t index) const;
    // moments are k-2 degrees
    PoissonVEM2(const VEMMesh2 &_mesh, size_t max_cell_degrees);
    PoissonVEM2(const VEMMesh2 &_mesh, size_t max_cell_degrees,
                size_t edge_subsamples);
    PoissonVEM2(const VEMMesh2 &_mesh, std::vector<size_t> max_degrees,
                std::vector<size_t> edge_subsamples,
                std::vector<size_t> moment_degrees,
                std::vector<double> diameters = {});

    // for a mesh using kth order elements, relative_order=N returns a mesh
    // using k-N cells typically you should use k-1 for pressure values
    PoissonVEM2 relative_order_mesh(int relative_order) const;
    mtao::VecXd active_cell_polynomial_mask(
        const std::set<int> &used_cells = {}) const;

    Eigen::SparseMatrix<double> stiffness_matrix(
        const std::set<int> &used_cells = {}) const;

    // Pi^*
    Eigen::SparseMatrix<double> mass_matrix(
        const std::set<int> &used_cells = {}) const;

    // L2(m,p)
    // Eigen::SparseMatrix<double> mass_matrix(
    //    const std::set<int>& used_cells = {}) const;

    // should be deprecated, it's set to point_laplacian though
    Eigen::SparseMatrix<double> laplacian(
        const std::set<int> &used_cells = {}) const;

    // Pi^* G^* diag(M) G Pi
    Eigen::SparseMatrix<double> point_laplacian(
        const std::set<int> &used_cells = {}) const;

    // G^* diag(Pi^* M Pi) G
    Eigen::SparseMatrix<double> poly_laplacian(
        const std::set<int> &used_cells = {}) const;
    // Eigen::SparseMatrix<double> stiffness_matrix_sqrt(
    //    const std::set<int>& used_cells = {}) const;

    // diag(K_{pq}) * G
    Eigen::SparseMatrix<double> poly_to_sample_gradient(
        const std::set<int> &used_cells = {}) const;
    // G * Pi
    Eigen::SparseMatrix<double> sample_to_poly_gradient(
        const std::set<int> &used_cells = {}) const;
    // diag(K_{pq}) * G * Pi
    Eigen::SparseMatrix<double> sample_to_sample_gradient(
        const std::set<int> &used_cells = {}) const;
    // G^* * diag(Pi)
    Eigen::SparseMatrix<double> sample_to_poly_divergence(
        const std::set<int> &used_cells = {}) const;
    // K_{pq} * G^* * diag(Pi)
    Eigen::SparseMatrix<double> sample_to_sample_divergence(
        const std::set<int> &used_cells = {}) const;

    // G * Pi
    Eigen::SparseMatrix<double> sample_to_poly_lap_gradient(
        const std::set<int> &used_cells = {}) const;
    // diag(K_{pq}) G Pi + I - K_{pq}Pi
    Eigen::SparseMatrix<double> sample_to_sample_lap_gradient(
        const std::set<int> &used_cells = {}) const;
    // Pi^* G^* M
    Eigen::SparseMatrix<double> poly_to_sample_lap_cogradient(
        const std::set<int> &used_cells = {}) const;
    // Pi^* G^* M diag(Pi) + (I - K_{pq}Pi)^*(I - K_{pq}Pi)
    Eigen::SparseMatrix<double> sample_to_sample_lap_cogradient(
        const std::set<int> &used_cells = {}) const;

    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> kkt_system(
        const ScalarConstraints &constraints,
        const std::set<int> &used_cells = {}) const;

    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f) const;
    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f,
        int samples_per_cell) const;
    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles = {}) const;

    // writes constraints from monomial coefficients to satisfy a
    // scalarconstraint object
    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
    point_constraint_matrix(const ScalarConstraints &constraints,
                            const std::set<int> &used_cells = {}) const;
    std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
    polynomial_constraint_matrix(const ScalarConstraints &constraints,
                                 const std::set<int> &used_cells = {}) const;

    // World space points to per-cell polynomials
    Eigen::SparseMatrix<double> sample_to_polynomial_projection_matrix(
        const std::set<int> &used_cells = {}) const;

    Eigen::SparseMatrix<double> projection_error(
        const std::set<int> &used_cells = {}) const;

    // Equivaelnt to K_{pq} for all cells while guaranteeing that un-broken
    // polynomials are evaluted correctly
    Eigen::SparseMatrix<double> polynomial_to_sample_evaluation_matrix(
        CellWeightWeightMode mode = CellWeightWeightMode::AreaWeighted,
        const std::set<int> &used_cells = {}) const;

    int system_size() const;
    int point_size() const;
    int moment_size() const;
    int monomial_size() const;
    int cell_count() const;

    const MonomialBasisIndexer &monomial_indexer() const {
        return _monomial_indexer;
    }

    const MomentBasisIndexer &moment_indexer() const { return _moment_indexer; }
    const PointSampleIndexer &point_sample_indexer() const {
        return _point_sample_indexer;
    }
    const VEMMesh2 &mesh() const { return _mesh; }
    std::shared_ptr<VEMMesh2 const> mesh_ptr() const { return _mesh.handle(); }

   private:
    const VEMMesh2 &_mesh;

    // the number of subsamples applied on each edge
    std::vector<size_t> _edge_internal_samples;
    // per cell monomial degrees
    std::vector<size_t> _monomial_degrees;

    // all indexers assume their first idnex is 0
    PointSampleIndexer _point_sample_indexer;
    MonomialBasisIndexer _monomial_indexer;
    MomentBasisIndexer _moment_indexer;
};
}  // namespace vem::poisson_2d
