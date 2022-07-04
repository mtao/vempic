#include "vem/serialization/serialize_particles.hpp"

#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/point_cloud/partio_loader_impl.hpp>

namespace vem::serialization {

void serialize_particles_with_velocities(Inventory& inventory,
                                         const std::string& name,
                                         const mtao::ColVecs6d& particles) {
    auto p = inventory.get_new_asset_path(name, "bgeo.gz");

    mtao::geometry::point_cloud::PartioFileWriter w(p);
    w.set_positions(particles.topRows<3>());
    w.set_velocities(particles.bottomRows<3>());

    inventory.asset_metadata(name)["storage_type"] = "partio";
}
void serialize_particles(Inventory& inventory, const std::string& name,
                         const mtao::ColVecs3d& particles) {
    auto p = inventory.get_new_asset_path(name, "bgeo.gz");
    mtao::geometry::point_cloud::PartioFileWriter w(p);
    w.set_positions(particles);

    inventory.asset_metadata(name)["storage_type"] = "partio";
}

mtao::ColVecs6d deserialize_particles_with_velocities(
    const Inventory& inventory, const std::string& name) {
    std::string type = inventory.asset_metadata(name).at("storage_type");
    if (type == "partio") {
        mtao::geometry::point_cloud::PartioFileReader r(
            inventory.get_asset_path(name));
        return mtao::eigen::vstack(r.positions(), r.velocities());
    } else {
        return deserialize_points6(inventory, name);
    }
}
mtao::ColVecs3d deserialize_particles(Inventory& inventory,
                                      const std::string& name) {
    std::string type = inventory.asset_metadata(name).at("storage_type");
    if (type == "partio") {
        mtao::geometry::point_cloud::PartioFileReader r(
            inventory.get_asset_path(name));
        return r.positions();
    } else {
        return deserialize_points3(inventory, name);
    }
}

void serialize_particles_with_velocities_and_densities(
    Inventory& inventory, const std::string& name,
    const mtao::ColVecs6d& particles, const mtao::VecXd& densities) {
    auto p = inventory.get_new_asset_path(name, "bgeo.gz");

    mtao::geometry::point_cloud::PartioFileWriter writer(p);
    writer.set_attribute("position", particles.topRows<3>().cast<float>());
    writer.set_attribute("velocity", particles.bottomRows<3>().cast<float>());
    writer.set_attribute("density", densities.cast<float>());
}

std::tuple<mtao::ColVecs6d, mtao::VecXd>
deserialize_particles_with_velocities_and_densities(const Inventory& inventory,
                                                    const std::string& name) {
    auto path = inventory.get_asset_path(name);
    mtao::geometry::point_cloud::PartioFileReader r(path);
    return {mtao::eigen::vstack(r.positions(), r.velocities()),
            r.attribute<float>("density").cast<double>()};
}
}  // namespace vem::serialization
