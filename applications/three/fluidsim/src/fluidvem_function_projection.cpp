#define EIGEN_DONT_PARALLELIZE

#include <tbb/parallel_for.h>

#include <chrono>
#include <thread>
#include <vem/three/boundary_facets.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/loop_over_active.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/three/fluidsim/fluidvem.hpp"

using namespace std::chrono_literals;
namespace vem::three::fluidsim {

mtao::VecXd FluidVEM3::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec3d &)> &f) const {
    double val = (double)(pressure_monomial_size()) / cell_count() + 3;
    return coefficients_from_point_sample_function(f, val * val);
}
mtao::VecXd FluidVEM3::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec3d &)> &f,
    int samples_per_cell) const {
    auto [P, O] = sample_active_cells(samples_per_cell);
    return coefficients_from_point_sample_function(f, P, O);
}
mtao::VecXd FluidVEM3::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec3d &)> &f,
    const mtao::ColVecs3d &P,
    const std::vector<std::set<int>> &cell_particles) const {

    spdlog::info("FluidVEM3::coefficients_from_point_sample_function");
    Eigen::setNbThreads(1);
    if (cell_particles.size() == 0) {
        auto new_cell_particles =
            utils::CellIdentifier<VEMMesh3>{mesh()}.cell_ownerships(P);
        if (!new_cell_particles.empty()) {
            return coefficients_from_point_sample_function(f, P,
                                                           new_cell_particles);
        } else {
            return {};
        }
    }
    mtao::VecXd A(pressure_sample_size());

    auto face_cob = face_coboundary_map(mesh(), active_cells());

    tbb::parallel_for(int(0), mesh().face_count(), [&](int face_index) {
        auto samples = pressure_weighted_face_samples(face_index);
        const auto &cob = face_cob.at(face_index);
        int sample_count = cob.size() * samples.cols();
        auto indices =
            pressure_indexer().flux_indexer().coefficient_indices(face_index);
        if (indices.size() == 0) {
            return;
        }

        auto pblock = A.segment(*indices.begin(), indices.size());

        pblock.setZero();
        for (auto &&cell_index : cob) {
            auto c = get_pressure_cell(cell_index);
            for (int j = 0; j < samples.cols(); ++j) {
                auto pt = samples.col(j).head<3>();
                double v = f(pt);
                pblock += c.evaluate_monomials_by_size(pblock.size(), pt) * v;
            }
        }
        pblock /= sample_count;
    });

    tbb::parallel_for(size_t(0), cell_particles.size(), [&](size_t cell_index) {
        const auto &particles = cell_particles[cell_index];

        const auto &momi = pressure_indexer().moment_indexer();
        auto c = get_pressure_cell(cell_index);
        size_t mom_size = c.moment_size();
        if (mom_size == 0) {
            return;
        }
        size_t mom_offset = c.global_moment_index_offset();

        auto pblock = A.segment(mom_offset, mom_size);

        pblock.setZero();

        auto run = [&](auto &&pt) {
            double v = f(pt);
            pblock += c.evaluate_monomials_by_size(pblock.size(), pt) * v;
        };

        for (auto &&p : particles) {
            run(P.col(p));
        }

        // auto LQFit =
        //    l2c.unweighted_least_squares_coefficients(max_degree, LP, V);

        pblock /= particles.size();
        // bpblock = (c.monomial_l2_grammian() *
        // LQFit).head(c.moment_size()) /
        //          c.volume();
    });

    Eigen::setNbThreads(0);
    return A;
}

mtao::ColVecs3d FluidVEM3::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f) const {
    double val = (double)(velocity_stride_monomial_size()) / cell_count() + 3;
    return coefficients_from_point_sample_vector_function(f, val * val);
}
mtao::ColVecs3d FluidVEM3::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f,
    int samples_per_cell) const {
    spdlog::info("coeff from point sample vector function sample");
    auto [P, O] = sample_active_cells(samples_per_cell);
    return coefficients_from_point_sample_vector_function(f, P, O);
}

mtao::ColVecs3d FluidVEM3::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec3d(const mtao::Vec3d &)> &f,
    const mtao::ColVecs3d &P,
    const std::vector<std::set<int>> &cell_particles) const {
    spdlog::info("FluidVEM3::coefficients_from_point_sample_vector_function");
    Eigen::setNbThreads(1);
    if (cell_particles.size() == 0) {
        auto new_cell_particles =
            utils::CellIdentifier<VEMMesh3>{mesh()}.cell_ownerships(P);
        if (!new_cell_particles.empty()) {
            return coefficients_from_point_sample_vector_function(
                f, P, new_cell_particles);
        } else {
            return {};
        }
    }
    //spdlog::info("Sleeping for 5 secs after getting cell identification");
    //std::this_thread::sleep_for(5s);
    mtao::ColVecs3d R(3, velocity_stride_sample_size());

    int face_index = 0;
    auto face_cob = face_coboundary_map(mesh(), active_cells());
    //spdlog::info("Sleeping for 5 secs for getting cob");
    //std::this_thread::sleep_for(5s);

    tbb::parallel_for(int(0), mesh().face_count(), [&](int face_index) {
        // for (face_index = 0; face_index < mesh().face_count(); ++face_index)
        // {
        return;
        auto samples = velocity_weighted_face_samples(face_index);
        // std::cout << "samples:\n";
        // std::cout << samples << std::endl;
        const auto &cob = face_cob.at(face_index);
        int sample_count = cob.size() * samples.cols();
        auto indices =
            velocity_indexer().flux_indexer().coefficient_indices(face_index);
        if (indices.size() == 0) {
            return;
        }

        auto pblock = R.block(0, *indices.begin(), 3, indices.size());

        pblock.setZero();
        for (auto &&cell_index : cob) {
            auto c = get_velocity_cell(cell_index);

            for (int j = 0; j < samples.cols(); ++j) {
                auto pt = samples.col(j).head<3>();
                mtao::Vec3d v = f(pt);
                auto mom = c.evaluate_monomials_by_size(pblock.cols(), pt);
                pblock += v * mom.transpose();
            }
        }
        pblock /= sample_count;
    });
    //spdlog::info("Sleeping for 5 secs after getting face data");
    //std::this_thread::sleep_for(5s);

    tbb::parallel_for(size_t(0), cell_particles.size(), [&](size_t cell_index) {
        // spdlog::info("coeffs from point vec func: Moment cell {}",
        // cell_index);
        const auto &particles = cell_particles[cell_index];

        const auto &momi = velocity_indexer().moment_indexer();

        auto c = get_velocity_cell(cell_index);
        size_t mom_size = c.moment_size();
        if (mom_size == 0) {
            return;
        }
        size_t mom_offset = c.global_moment_index_offset();

        auto pblock = R.block(0, mom_offset, 3, mom_size);

        pblock.setZero();

        auto run = [&](auto &&pt) {
            auto v = f(pt);
            pblock +=
                v * c.evaluate_monomials_by_size(pblock.cols(), pt).transpose();
        };

        for (auto &&p : particles) {
            run(P.col(p));
        }

        pblock /= particles.size();
    });
    //spdlog::info("Sleeping for 5 secs after getting cell data");
    //std::this_thread::sleep_for(5s);
    Eigen::setNbThreads(0);
    return R;
}

std::tuple<mtao::ColVecs3d, std::vector<std::set<int>>>
FluidVEM3::sample_active_cells(size_t samples_per_cell) const {
    spdlog::info("FluidVEM3::coefficients_from_point_sample_vector_function");
    Eigen::setNbThreads(1);
    std::vector<std::set<int>> ownerships(cell_count());
    mtao::ColVecs3d points(3, samples_per_cell * cell_count());
    tbb::parallel_for(size_t(0), ownerships.size(), [&](size_t idx) {
        // for (auto &&[idx, own] : mtao::iterator::enumerate(ownerships)) {
        auto &own = ownerships[idx];
        if (is_active_cell(idx)) {
            auto c = get_velocity_cell(idx);
            auto bb = c.bounding_box();
            int offset = idx * samples_per_cell;
            for (int j = 0; j < samples_per_cell; ++j) {
                own.emplace(j + offset);
                auto p = points.col(j + offset) = bb.sample();
                while (!c.is_inside(p)) {
                    p = bb.sample();
                }
            }
        }
    });
    Eigen::setNbThreads(0);
    return {points, ownerships};
}

}  // namespace vem::fluidsim_3d
