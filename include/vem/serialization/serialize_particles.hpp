
#pragma once

#include <mtao/types.hpp>

#include "vem/serialization/inventory.hpp"
#include "vem/serialization/serialize_eigen.hpp"
namespace vem::serialization {

void serialize_particles_with_velocities(Inventory& inventory,
                                         const std::string& name,
                                         const mtao::ColVecs6d& particles);

void serialize_particles_with_velocities_and_densities(
    Inventory& inventory, const std::string& name,
    const mtao::ColVecs6d& particles, const mtao::VecXd& densities);
void serialize_particles(Inventory& inventory, const std::string& name,
                         const mtao::ColVecs3d& particles);

mtao::ColVecs3d deserialize_particles(Inventory& inventory,
                                      const std::string& name);

mtao::ColVecs6d deserialize_particles_with_velocities(
    const Inventory& inventory, const std::string& name);

std::tuple<mtao::ColVecs6d, mtao::VecXd>
deserialize_particles_with_velocities_and_densities(const Inventory& inventory,
                                                    const std::string& name);
}  // namespace vem::serialization
