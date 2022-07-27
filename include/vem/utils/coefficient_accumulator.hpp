#pragma once
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/utils/loop_over_active.hpp>


namespace vem::utils{
template <typename IndexerType>
struct CoefficientAccumulator {
    const IndexerType &indexer;
    const auto &mesh() const { return indexer.mesh(); }
    using RBFFunc =
        std::function<double(const mtao::Vec2d &a, const mtao::Vec2d &b)>;

    CoefficientAccumulator(const IndexerType &a) : indexer(a) {}

    // create the boundary coefficients
    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_boundary_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &sf) const;
    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_boundary_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_interior_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    // just creat ethe internal coefficients
    mtao::ColVectors<double, D + 1>
    homogeneous_interior_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_interior_coefficients_from_point_function(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    // the general functions that creat all coefficients

    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
        auto B = homogeneous_boundary_coefficients_from_point_function<D>(
            f, P, cell_particles, active_cells);
        // std::cout << "vvvvvvvvvvvvvv" << std::endl;
        auto M = homogeneous_interior_coefficients_from_point_function<D>(
            f, P, cell_particles, active_cells);
        // std::cout << M << std::endl;
        // std::cout << "^^^^^^^^^^^^^^" << std::endl;
        return mtao::eigen::hstack(B, M);
    }

    template <int D>
    mtao::ColVectors<double, D + 1> homogeneous_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &sf) const {
        auto B = homogeneous_boundary_coefficients_from_point_values<D>(
            V, P, cell_particles, active_cells, sf);
        auto M = homogeneous_interior_coefficients_from_point_values<D>(
            V, P, cell_particles, active_cells);
        return mtao::eigen::hstack(B, M);
    }
};

template <typename IndexerType>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<IndexerType>::
    homogeneous_interior_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.moment_size());
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    tbb::parallel_for(size_t(0), size_t(cell_particles.size()), [&](size_t cell_index) {
        const auto &particles = cell_particles.at(cell_index);
        // for (auto&& [cell_index, particles] :
        //     mtao::iterator::enumerate(cell_particles)) {
        auto c = indexer.get_cell(cell_index);

        if (c.moment_size() == 0) {
            return;
        }

        auto pblock = Dat.block(0, c.moment_only_global_moment_index_offset(),
                                D, c.moment_size());
        auto wblock = W.segment(c.moment_only_global_moment_index_offset(),
                                c.moment_size());
        pblock.setZero();
        wblock.setConstant(double(particles.size()));
        auto run = [&](auto &&pt, auto &&v) {
            auto block = c.evaluate_monomials_by_size(pblock.cols(), pt);
            pblock += v * block.transpose();
        };

        for (auto &&p : particles) {
            auto pos = P.col(p);
            run(pos, V.col(p));
        }
    });
    return R;
}

template <typename IndexerType>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<IndexerType>::
    homogeneous_interior_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.moment_size());
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    int cell_index = 0;
    tbb::parallel_for(size_t(0), cell_particles.size(), [&](size_t cell_index) {
        const auto &particles = cell_particles.at(cell_index);
        // spdlog::info("Cell {} has particles {}", cell_index,
        //             fmt::join(particles, ","));
        // for (auto&& [cell_index, particles] :
        //     mtao::iterator::enumerate(cell_particles)) {
        auto c = indexer.get_cell(cell_index);

        if (c.moment_size() == 0) {
            return;
        }

        auto pblock = Dat.block(0, c.moment_only_global_moment_index_offset(),
                                D, c.moment_size());
        auto wblock = W.segment(c.moment_only_global_moment_index_offset(),
                                c.moment_size());
        pblock.setZero();
        wblock.setConstant(double(particles.size()));
        auto run = [&](auto &&pt, auto &&v) {
            auto block = c.evaluate_monomials_by_size(pblock.cols(), pt);
            pblock += v * block.transpose();
        };

        for (auto &&p : particles) {
            auto pos = P.col(p);
            // std::cout << pos.transpose() << std::endl;
            run(pos, f(pos));
            // std::cout << "Updated pblock:\n" << pblock << std::endl;
        }
    });

    return R;
}


}  // namespace vem::utils
