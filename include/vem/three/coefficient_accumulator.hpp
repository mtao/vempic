#pragma once
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <vem/serialization/serialize_eigen.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "flux_moment_indexer.hpp"
#include "mesh.hpp"
#include "boundary_facets.hpp"
#include "cells_adjacent_to_face.hpp"

namespace vem::three {
struct CoefficientAccumulator3 {
    const FluxMomentIndexer3 &indexer;
    const auto &mesh() const { return indexer.mesh(); }
    using RBFFunc =
        std::function<double(const mtao::Vec3d &a, const mtao::Vec3d &b)>;

    CoefficientAccumulator3(const FluxMomentIndexer3 &a) : indexer(a) {}

    // create the boundary coefficients
    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_boundary_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &sf) const;
    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_boundary_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_interior_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    // just creat ethe internal coefficients
    mtao::ColVectors<double, D + 1>
    homogeneous_interior_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_interior_coefficients_from_point_function(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const;

    // the general functions that creat all coefficients

    template <int D>
    mtao::ColVectors<double, D + 1>
    homogeneous_coefficients_from_point_function(
        const std::function<
            typename mtao::Vector<double, D>(const mtao::Vec3d &)> &f,
        const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells) const {
        auto B = homogeneous_boundary_coefficients_from_point_function<D>(
            f, P, cell_particles, active_cells);
        auto M = homogeneous_interior_coefficients_from_point_function<D>(
            f, P, cell_particles, active_cells);
        auto R = mtao::eigen::hstack(B, M);
        return R;
    }

    template <int D>
    mtao::ColVectors<double, D + 1> homogeneous_coefficients_from_point_values(
        const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
        const std::vector<std::set<int>> &cell_particles,
        const std::set<int> &active_cells, const RBFFunc &sf) const {
        auto B = homogeneous_boundary_coefficients_from_point_values<D>(
            V, P, cell_particles, active_cells, sf);
        auto M = homogeneous_interior_coefficients_from_point_values<D>(
            V, P, cell_particles, active_cells);
        auto R = mtao::eigen::hstack(B, M);
        return R;
    }
};

template <int D>
mtao::ColVectors<double, D + 1>
CoefficientAccumulator3::homogeneous_interior_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    spdlog::info("CoefficientAccumulator3::homogeneous_interior_coefficients_from_point_values");

    Eigen::setNbThreads(1);
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.moment_size());
    if (R.cols() == 0) {
        return R;
    }
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    int cell_index = 0;
    tbb::parallel_for(
        size_t(0), size_t(cell_particles.size()), [&](size_t cell_index) {
            const auto &particles = cell_particles.at(cell_index);
            // for (auto&& [cell_index, particles] :
            //     mtao::iterator::enumerate(cell_particles)) {
            auto c = indexer.get_cell(cell_index);

            if (c.moment_size() == 0) {
                return;
            }

            auto pblock =
                Dat.block(0, c.moment_only_global_moment_index_offset(), D,
                          c.moment_size());
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
    Eigen::setNbThreads(0);
    return R;
}

template <int D>
mtao::ColVectors<double, D + 1>
CoefficientAccumulator3::homogeneous_interior_coefficients_from_point_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec3d &)>
        &f,
    const mtao::ColVecs3d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {

    spdlog::info("CoefficientAccumulator3::homogeneous_interior_coefficients_from_point_function");
    Eigen::setNbThreads(1);
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.moment_size());
    if (R.cols() == 0) {
        return R;
    }
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    tbb::parallel_for(
        size_t(0), size_t(cell_particles.size()), [&](size_t cell_index) {
            const auto &particles = cell_particles.at(cell_index);
            // spdlog::info("Cell {} has particles {}", cell_index,
            //             fmt::join(particles, ","));
            // for (auto&& [cell_index, particles] :
            //     mtao::iterator::enumerate(cell_particles)) {
            auto c = indexer.get_cell(cell_index);

            if (c.moment_size() == 0) {
                return;
            }

            auto pblock =
                Dat.block(0, c.moment_only_global_moment_index_offset(), D,
                          c.moment_size());
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

    Eigen::setNbThreads(0);
    return R;
}

template <int D>
mtao::ColVectors<double, D + 1>
CoefficientAccumulator3::homogeneous_boundary_coefficients_from_point_values(
    const mtao::ColVectors<double, D> &V, const mtao::ColVecs3d &P,
    const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells, const RBFFunc &rbf) const {
    spdlog::info("CoefficientAccumulator3::homogeneous_boundary_coefficients_from_point_values");
    Eigen::setNbThreads(1);
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());
    spdlog::info("Writing homogeneous boundary coefficients from piont values");

    R.setZero();
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);

    auto face_cob = face_coboundary_map(mesh(), active_cells);
    auto face_cell_neighbors =
        cells_adjacent_to_face(mesh(), active_cells);
    // mtao::vector<mtao::Vec4d> backprojections;
    tbb::parallel_for(
        size_t(0), size_t(mesh().face_count()), [&](size_t face_index) {
            if (indexer.is_face_inactive(face_index)) {
                return;
            }
            //spdlog::info("Internal loop {} / {}", face_index, mesh().face_count());
            mtao::ColVecs4d samples = indexer.weighted_face_samples(face_index);
            mtao::ColVecs2d samples_planar =
                indexer.flux_indexer().point_to_st(face_index) *
                samples.topRows<3>().colwise().homogeneous();
            const auto &cob = face_cob.at(face_index);
            const auto &cell_neighbors = face_cell_neighbors.at(face_index);
            int sample_count = cob.size() * samples.cols();
            auto indices =
                indexer.flux_indexer().coefficient_indices(face_index);
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
                        auto pt = samples.col(j).head<3>();
                        auto st = samples_planar.col(j);

                        double weight =
                            rbf(pposition,
                                pt);  // * (p - center).normalized();

                        // std::cout << "Weight: " << weight
                        //          << "(d = " << (pposition - pt).norm() << ")"
                        //          << std::endl;
                        auto moms = indexer.flux_indexer()
                                        .evaluate_monomials_by_size_local(
                                            pblock.cols(), st);
                        pblock += weight * val * moms.transpose();
                        // std::cout << wblock << " <=== " << weight <<
                        // std::endl;
                        wblock.array() += weight;
                        //    * v;
                    }
                }
            }
        });

    Eigen::setNbThreads(0);
    return R;
}

template <int D>
mtao::ColVectors<double, D + 1>
CoefficientAccumulator3::homogeneous_boundary_coefficients_from_point_function(
    const std::function<typename mtao::Vector<double, D>(const mtao::Vec3d &)>
        &f,
    const mtao::ColVecs3d &P, const std::vector<std::set<int>> &cell_particles,
    const std::set<int> &active_cells) const {
    spdlog::info("CoefficientAccumulator3::homogeneous_boundary_coefficients_from_point_function");
    Eigen::setNbThreads(0);
    mtao::ColVectors<double, D + 1> R(D + 1, indexer.boundary_size());
    R.setZero();
    auto Dat = R.template topRows<D>();
    auto W = R.row(D);
    auto face_cob = face_coboundary_map(mesh(), active_cells);
    auto face_cell_neighbors =
        cells_adjacent_to_face(mesh(), active_cells);
    //std::mutex mut;
    tbb::parallel_for(
        size_t(0), size_t(mesh().face_count()), [&](size_t face_index) {
        //std::scoped_lock sl(mut);
            if (indexer.is_face_inactive(face_index)) {
                return;
            }
            //spdlog::info("homo bound coeffs from point func: Face index: {}", face_index);
            //std::cout << "Normal: " << mesh().normal(face_index).transpose() << std::endl;
            //std::cout << "Center: " << mesh().FC.col(face_index).transpose() << std::endl;
            auto samples = indexer.weighted_face_samples(face_index);
            //std::cout << "Samples\n" << samples << std::endl;
            mtao::ColVecs2d samples_planar =
                indexer.flux_indexer().point_to_st(face_index) *
                samples.topRows<3>().colwise().homogeneous();
            //std::cout << "Samples in plane\n" << samples_planar << std::endl;
            const auto &cell_neighbors = face_cell_neighbors.at(face_index);
            auto indices =
                indexer.flux_indexer().coefficient_indices(face_index);
            if (indices.size() == 0) {
                return;
            }

            auto pblock = Dat.block(0, *indices.begin(), D, indices.size());
            auto wblock = W.segment(*indices.begin(), indices.size());

            pblock.setZero();
            double weight_sum = 0;

            for (int j = 0; j < samples.cols(); ++j) {
                auto s = samples.col(j);
                auto st = samples_planar.col(j);
                auto pt = s.template head<3>();
                double w = s(3);
                auto v = f(pt);
                auto mom =
                    indexer.flux_indexer().evaluate_monomials_by_size_local(
                        pblock.cols(), st);
                auto val = w * v * mom.transpose();
                pblock += val;
                weight_sum += w;// * std::pow(indexer.flux_indexer().diameter(face_index);

                // pblock /= sample_count;
            }
            wblock.setConstant(weight_sum);
        });
    Eigen::setNbThreads(1);
    return R;
}
}  // namespace vem::utils
