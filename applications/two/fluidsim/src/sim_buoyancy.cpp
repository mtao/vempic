
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/solvers/linear/conjugate_gradient.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <vem/two/normals.hpp>
#include <vem/two/parent_maps.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/two/fluidsim/sim.hpp"

namespace vem::two::fluidsim{

void Sim::add_buoyancy_force(double dt) {
    sample_velocities.row(1) += dt * sample_density;
    spdlog::info("Added a buoyancy force of {}", dt * sample_density.norm());
}
void Sim::update_buoyancy_particles_with_source() {
    auto bb = mesh().bounding_box();

    double bottom = bb.min().y() + .1 * bb.sizes().y();
    double radius = .1 * bb.sizes().x();
    double center = bb.center().x();

    if (particle_density.size() != num_particles()) {
        particle_density.resize(num_particles());
        particle_density.setConstant(0);
    }

    auto density_source_region = [&](auto&& p) -> bool {
        if (p.y() < bottom && std::abs(p.x() - center) < radius) {
        //if((p - bb.center()).norm() < radius) {

            return true;
        } else {
            return false;
        }
    };

    for (int j = 0; j < num_particles(); ++j) {
        if (density_source_region(particle_position(j))) {
            particle_density(j) = 1;
        }
    }
}
}  // namespace vem::fluidsim_2d
