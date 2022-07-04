#pragma once
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/utils/boundary_facets.hpp>
#include <vem/utils/cells_adjacent_to_edge.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "vem/flux_moment_indexer.hpp"
#include "vem/mesh.hpp"
#include "vem/point_moment_indexer.hpp"

namespace vem::utils {
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

template <>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<PointMomentIndexer>::
    homogeneous_boundary_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &rbf) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());

    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    W.setZero();
    auto B = Dat.leftCols(W.size());
    B.setZero();
    for (auto &&[cell_index, particles] :
         mtao::iterator::enumerate(cell_particles)) {
        auto c = indexer.get_cell(cell_index);
        for (auto &&point_sample_index : c.point_sample_indices()) {
            auto s =
                indexer.point_sample_indexer().get_position(point_sample_index);
            auto sample_vel = Dat.col(point_sample_index);
            double &weight_sum = W(point_sample_index);
            for (auto &&p : particles) {
                double weight =
                    rbf(s, P.col(p));  // * (p - center).normalized();

                weight_sum += weight;
                sample_vel += weight * V.col(p);
            }
        }
    }
    return R;
}
template <>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<FluxMomentIndexer>::
    homogeneous_boundary_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &rbf) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());

    R.setZero();
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    auto edge_cob = utils::edge_coboundary_map(mesh(), active_cells);
    auto edge_cell_neighbors =
        utils::cells_adjacent_to_edge(mesh(), active_cells);
    int edge_index = 0;
    // mtao::vector<mtao::Vec4d> backprojections;
    tbb::parallel_for(
        int(edge_index), int(mesh().edge_count()), [&](int edge_index) {
            if (indexer.is_edge_inactive(edge_index)) {
                return;
            }
            auto samples = indexer.weighted_edge_samples(edge_index);
            const auto &cob = edge_cob.at(edge_index);
            const auto &cell_neighbors = edge_cell_neighbors.at(edge_index);
            int sample_count = cob.size() * samples.cols();
            auto indices =
                indexer.boundary_indexer().coefficient_indices(edge_index);
            if (indices.size() == 0) {
                return;
            }
            const bool is_interior = cob.size() == 2;

            auto pblock = Dat.block(0, *indices.begin(), D, indices.size());
            auto wblock = W.segment(*indices.begin(), indices.size());

            for (auto &&cell_index : cell_neighbors) {
                const auto &particles = cell_particles[cell_index];
                auto c = indexer.get_cell(cell_index);
                for (auto &&p : particles) {
                    auto pposition = P.col(p);
                    auto val = V.col(p);
                    // std::cout << pvelocity.transpose() << std::endl;
                    // std::cout << pposition.transpose() << " => " << val
                    //          << std::endl;

                    for (int j = 0; j < samples.cols(); ++j) {
                        auto pt = samples.col(j).head<2>();

                        double weight =
                            rbf(pposition,
                                pt);  // * (p - center).normalized();

                        // std::cout << "Weight: " << weight
                        //          << "(d = " << (pposition - pt).norm() << ")"
                        //          << std::endl;
                        auto moms =
                            c.evaluate_monomials_by_size(pblock.cols(), pt);
                        pblock += weight * val * moms.transpose();
                        // std::cout << wblock << " <=== " << weight <<
                        // std::endl;
                        wblock.array() += weight;
                        //    * v;
                    }
                }
            }
        });

    return R;
}
template <>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<PointMomentIndexer>::
    homogeneous_boundary_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());
    R.setZero();
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);
    W.setConstant(1);
    tbb::parallel_for(size_t(0), indexer.point_sample_size(), [&](size_t j) {
        mtao::Vec2d p = indexer.point_sample_indexer().get_position(j);
        Dat.col(j) = f(p);
    });
    return R;
}

template <>
template <int D>
mtao::ColVectors<double, D + 1> CoefficientAccumulator<FluxMomentIndexer>::
    homogeneous_boundary_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec2d &)> &f,
        const mtao::ColVecs2d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());
    R.setZero();
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);
    auto edge_cob = utils::edge_coboundary_map(mesh(), active_cells);
    tbb::parallel_for(
        int(0), int(mesh().edge_count()), [&](int edge_index) {
            if (indexer.is_edge_inactive(edge_index)) {
                return;
            }
            auto samples = indexer.weighted_edge_samples(edge_index);
            const auto &cob = edge_cob.at(edge_index);
            int sample_count = cob.size() * samples.cols();
            auto indices =
                indexer.boundary_indexer().coefficient_indices(edge_index);
            if (indices.size() == 0) {
                return;
            }

            auto pblock = Dat.block(0, *indices.begin(), D, indices.size());
            auto wblock = W.segment(*indices.begin(), indices.size());

            pblock.setZero();
            double weight_sum = 0;

            for (int j = 0; j < samples.cols(); ++j) {
                auto s = samples.col(j);
                auto pt = s.template head<2>();
                double w = s(2);
                auto v = f(pt);
                auto mom =
                    indexer.boundary_indexer().evaluate_monomials_by_size(
                        edge_index, pblock.cols(), pt);
                pblock += w * v * mom.transpose();
                weight_sum += w;

                // pblock /= sample_count;
            }
            wblock.setConstant(weight_sum);
        });
    return R;
}
}  // namespace vem::utils
