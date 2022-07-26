#pragma once
#include "flux_basis_indexer.hpp"
#include "flux_moment_cell.hpp"
#include "mesh.hpp"
#include "vem/monomial_basis_indexer.hpp"

namespace vem {
class FluxMomentIndexer3 {
   public:
    using MonomialIndexer = detail::MonomialBasisIndexer<3, 3>;
    using FluxIndexer = FluxBasisIndexer3;
    using MomentIndexer = detail::MonomialBasisIndexer<3, 3>;
    // this is passed by value to allow for moving of the data
    FluxMomentIndexer3(const VEMMesh3 &mesh, std::vector<size_t> max_degrees);
    FluxMomentIndexer3(const VEMMesh3 &_mesh, size_t max_degree);

    const MomentIndexer &moment_indexer() const { return _moment_indexer; }
    const FluxIndexer &flux_indexer() const { return _flux_indexer; }
    const FluxIndexer &boundary_indexer() const { return _flux_indexer; }

    const MonomialIndexer &monomial_indexer() const {
        return _monomial_indexer;
    }
    FluxMomentVEM3Cell get_cell(size_t index) const;

    size_t sample_size() const;
    size_t flux_size() const;
    size_t boundary_size() const { return flux_size(); }
    size_t moment_size() const;
    size_t monomial_size() const;

    bool is_face_inactive(int face_index) const;
    // an attempt at quadrature
    mtao::ColVecs4d weighted_face_samples_by_degree(int face_index,
                                                    int max_degree) const;
    mtao::ColVecs4d weighted_face_samples(int face_index,
                                          int sample_count) const;
    mtao::ColVecs4d weighted_face_samples(int face_index) const;

    mtao::ColVecs3d coefficients_from_point_sample_function(
        const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    mtao::VecXd coefficients_from_point_sample_function(
        const std::function<double(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    mtao::ColVecs3d coefficients_from_point_values(
        const mtao::ColVecs3d &V, const mtao::ColVecs3d &P,
        const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    mtao::VecXd coefficients_from_point_values(
        const mtao::VecXd &V, const mtao::ColVecs3d &P,
        const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells = {}) const;

    const VEMMesh3 &mesh() const { return _mesh; }

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

   private:
    const VEMMesh3 &_mesh;
    // all indexers assume their first idnex is 0
    FluxIndexer _flux_indexer;
    MomentIndexer _moment_indexer;
    // monomials have to come last because their degrees determine the prior 2
    // degrees
    MonomialIndexer _monomial_indexer;

    std::map<int, mtao::MatXd> _cached_l2_grammians;
    std::map<int, mtao::MatXd> _cached_regularized_dirichlet_grammians;

   private:
    // given just a bunch of points in space, construct a sample-based
    // representation implicitly uses some radius basis function type thing
    template <int D>
    mtao::ColVectors<double, D + 1> _homogeneous_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
        const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)>
            &rbf,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    // given a function and a bunch of points in space, construct a sample-based
    // representation
    template <int D>
    mtao::ColVectors<double, D + 1>
    _homogeneous_coefficients_from_point_sample_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;
    template <int D>
    mtao::ColVectors<double, D> _coefficients_from_point_sample_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    mtao::ColVectors<double, D> _coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
        const std::function<double(const mtao::Vec3d &, const mtao::Vec3d &)>
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
