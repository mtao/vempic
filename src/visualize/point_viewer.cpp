
#include "vem/visualize/point_viewer.hpp"

namespace vem::visualize {

PointViewer() {}
PointViewer(const Inventory& inv, const std::string& name)
    : AssetViewer(inv, name) {}
bool AssetViewer::load_implementation(const Inventory& inv,
                                      const std::string& name) {
    const auto& meta = my_inv.asset_metadata(name);
    std::string type = meta["type"];
    std::string storage_type = meta["storage_type"];
    if (type == "point2") {
        // particles

        auto V = serialization::deserialize_points2(my_inv, name)
                     .cast<float>()
                     .eval();
        point_data[name] = V;
    } else if (type == "point2,velocity2") {
        // particles with velocities, gotta split it up

        point_vector_data[name] =
            serialization::deserialize_points4(my_inv, name).cast<float>();
    } else if (type == "point2,velocity2,density1") {
    }
}

bool PointViewer::valid_type(const std::string& name) const {
    return valid_types().contains(name);
}
bool AssetViewer::valid_storage_type(const std::string& name) const {
    return valid_storage_types().contains(name);
}
std::set<std::string> PointViewer::valid_types() const {
    return {"point2", "point2,velocity2", "point2,density1",
            "point2,velocity2,density1"};
}

std::set<std::string> PointViewer::valid_storage_types() const {
    return {"market"};
}
}  // namespace vem::visualize

