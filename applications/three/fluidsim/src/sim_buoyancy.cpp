
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/logging/stopwatch.hpp>
#include <mtao/solvers/linear/conjugate_gradient.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/three/fluidsim/sim.hpp"

namespace vem::three::fluidsim {

void Sim::add_buoyancy_force(double dt) {
    spdlog::info("Adding buoyancy force");
    auto n = sample_velocities.rowwise().norm();

    sample_velocities.row(2) += dt * sample_density;
    velocity.coefficients().row(2) += dt * polynomial_density;
}
void Sim::update_buoyancy_particles_with_source() {
    spdlog::info("Updating buoyancy particles with source");
    auto bb = mesh().bounding_box();

    double bottom = bb.min().z() + .1 * bb.sizes().z();
    double radius = .05 * bb.sizes().head<2>().norm();
    mtao::Vec2d center = bb.center().head<2>();

    auto density_source_region = [&](auto&& p) -> bool {
        if (p.z() < bottom && (p.template head<2>() - center).norm() < radius) {
            return true;
        } else {
            return false;
        }
    };

#if defined(NO_TBB_FOR)
    for (int j = 0; j < particles.cols(); ++j) {
#else
    tbb::parallel_for<int>(0, particles.cols(), [&](int j) {
#endif
        bool doit = false;
        if (density_indicator_func) {
            doit = (*density_indicator_func)(particle_position(j));
        } else {
            doit = density_source_region(particle_position(j));
        }
        if (doit) {
            particle_density(j) = emitter_density;
        }
#if defined(NO_TBB_FOR)
    }
#else
    });
#endif
    update_buoyancy_from_particles();
}
void Sim::update_buoyancy_from_particles() {}
}  // namespace vem::fluidsim_3d
