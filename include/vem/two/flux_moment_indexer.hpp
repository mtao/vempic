#pragma once
#include "flux_basis_indexer.hpp"
#include "flux_moment_cell.hpp"
#include "mesh.hpp"
#include "vem/monomial_basis_indexer.hpp"

namespace vem::two {
class FluxMomentIndexer {
   public:
    // this is passed by value to allow for moving of the data
    FluxMomentIndexer(const VEMMesh2 &mesh, std::vector<size_t> max_degrees);
    FluxMomentIndexer(const VEMMesh2 &_mesh, size_t max_degree);

    const MomentBasisIndexer &moment_indexer() const { return _moment_indexer; }
    const FluxBasisIndexer &flux_indexer() const { return _flux_indexer; }
    const FluxBasisIndexer &boundary_indexer() const { return _flux_indexer; }

    const MonomialBasisIndexer &monomial_indexer() const {
        return _monomial_indexer;
    }
    FluxMomentVEM2Cell get_cell(size_t index) const;

    size_t sample_size() const;
    size_t flux_size() const;
    size_t boundary_size() const { return flux_size(); }
    size_t moment_size() const;
    size_t monomial_size() const;
    mtao::ColVecs3d weighted_edge_samples(int edge_index, int max_degree) const;
    mtao::ColVecs3d weighted_edge_samples(int edge_index) const;

    bool is_edge_inactive(int edge_index) const;

    mtao::ColVecs2d coefficients_from_point_sample_function(
        const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    mtao::ColVecs2d coefficients_from_point_values(
        const mtao::ColVecs2d &V, const mtao::ColVecs2d &P,
        const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    mtao::VecXd coefficients_from_point_values(
        const mtao::VecXd &V, const mtao::ColVecs2d &P,
        const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    const VEMMesh2 &mesh() const { return _mesh; }
    // Pi^* G^* diag(M) G Pi
    Eigen::SparseMatrix<double> sample_laplacian(
        const std::set<int> &active_cells = {}) const;
    Eigen::SparseMatrix<double> poly_laplacian(
        const std::set<int> &active_cells = {}) const;
    Eigen::SparseMatrix<double> sample_to_poly_l2(
        const std::set<int> &active_cells = {}) const;
    Eigen::SparseMatrix<double> sample_to_poly_dirichlet(
        const std::set<int> &active_cells = {}) const;

    Eigen::SparseMatrix<double> poly_l2_grammian(
        const std::set<int> &active_cells = {}) const;

    /*
    Eigen::SparseVector<double> inactive_face_sample_mask(
        const std::set<int> &inactive_faces) const;
    Eigen::SparseVector<double> inactive_face_sample_mask(
        const std::set<int> &inactive_faces) const;
    */

   private:
    const VEMMesh2 &_mesh;
    // all indexers assume their first idnex is 0
    FluxBasisIndexer _flux_indexer;
    MomentBasisIndexer _moment_indexer;
    // monomials have to come last because their degrees determine the prior 2
    // degrees
    MonomialBasisIndexer _monomial_indexer;

   private:
    // given just a bunch of points in space, construct a sample-based
    // representation implicitly uses some radius basis function type thing
    template <int D>
    mtao::ColVectors<double, D + 1> _homogeneous_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    // given a function and a bunch of points in space, construct a sample-based
    // representation
    template <int D>
    mtao::ColVectors<double, D + 1>
    _homogeneous_coefficients_from_point_sample_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;
    template <int D>
    mtao::ColVectors<double, D> _coefficients_from_point_sample_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    mtao::ColVectors<double, D> _coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::function<double(const mtao::Vec2d &, const mtao::Vec2d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;
    // Func should be mtao::MatXd(const FluidVEM2Cell&) where the returned is
    // poly x local_sample shaped
    template <typename Func>
    Eigen::SparseMatrix<double> sample_to_poly_cell_matrix(
        Func &&f, const std::set<int> &active_cells = {}) const;

    // Func should be mtao::MatXd(const FluidVEM2Cell&) where the returned is
    // local_sample x local_sample shaped
    template <typename Func>
    Eigen::SparseMatrix<double> sample_to_sample_cell_matrix(
        Func &&f, const std::set<int> &active_cells = {}) const;

    // Func should be mtao::MatXd(const FluidVEM2Cell&) where the returned is
    // local_poly x local_poly shaped
    template <typename Func>
    Eigen::SparseMatrix<double> poly_to_poly_cell_matrix(
        Func &&f, const std::set<int> &active_cells = {}) const;
};
}  // namespace vem
